# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import shutil, pathlib
import numpy as np
from types import SimpleNamespace

import Utility.timer

from ..mesh import marking, mesh_hierarchy
from ..linalg import solver_factory


#=================================================================#
class Model:
    def __init__(self, **kwargs):
        self.discretization_cls = kwargs.pop("discretization")
        self.discretization_params = kwargs.pop("discretization_params", {})
        # simple convenience: allow fem="cr1" directly
        if "fem" in kwargs:
            self.discretization_params["fem_name"] = kwargs.pop("fem")
        self.application = kwargs.pop("application")
        self.disc_params = kwargs.pop("disc_params", {})
        self.linear_solver = kwargs.pop("linear_solver", "spsolve")
        self.linear_solver_params = kwargs.pop("linear_solver_params", {})
        self.stack_storage = kwargs.pop("stack_storage", False)
        self.verbose = kwargs.pop("verbose",0)
        if kwargs:
            raise ValueError(f"unused arguments: {tuple(kwargs.keys())}")

        self.timer = Utility.timer.Timer()
        self.problemdata = self.application.problemdata

        datadir_def_name = f"{self.__class__.__name__}"+f"_{self.application.__class__.__name__}"
        if 'datadir_add' in kwargs:
            datadir_def_name += kwargs.pop('datadir_add')
        datadir_def =  pathlib.Path.home().joinpath( 'data_dir', datadir_def_name)
        self.datadir = kwargs.pop('datadir', datadir_def)
        if kwargs.pop("clean_data",True):
            try: shutil.rmtree(self.datadir)
            except: pass
        pathlib.Path(self.datadir).mkdir(parents=True, exist_ok=True)
        with open(self.datadir / "model", "w") as file:
            file.write(str(self))
        # check for unused arguments

        #--------prepare AFEM loop------
        self.mesh_hierarchy = None
        self.discs = []
        self.As = []
        self.transfers = []
        self.B = None
        self.init_afem()

    def init_afem(self):
        mesh0 = self.application.createMesh()

        self.mesh_hierarchy = mesh_hierarchy.MeshHierarchy(mesh0)
        self.discs = [self.discretize(mesh0)]

        self.As = []
        self.transfers = []

        self.B = solver_factory.getLinearSolver(
            method=self.linear_solver,
            As=self.As,
            transfers=self.transfers,
            **self.linear_solver_params,
        )

    def save(self, u, iter=None, datadir=None, name= "sol", add=''):
        if datadir is None: datadir=self.datadir
        if add: name += add
        if iter is not None: name += f"_{iter:05d}"
        np.save(datadir/name, u)
    def load(self, iter=None, datadir=None, name= "sol", add=''):
        if add: name += add
        if iter is not None: name += f"_{iter:05d}"
        if datadir is None: datadir=self.datadir
        name += ".npy"
        return np.load(datadir/name)
    #-------------------------------------------------------------------------------
    def afem_loop(self, niter, theta=0.9, plot_solution=False, plot_interpolation=False, mesh_timer=None):
        results = []

        for ell in range(niter):
            result = self.afem_step(
                ell,
                theta=theta,
                plot_solution=plot_solution,
                plot_interpolation=plot_interpolation,
                mesh_timer=mesh_timer,
            )
            results.append(result)

        return results

    def afem_step(self, ell, theta=0.9, plot_solution=False, plot_interpolation=False, mesh_timer=None):
        if plot_solution or plot_interpolation:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        disc = self.discs[-1]
        with self.timer.scope(f"AFEM{ell:02d}"):
            with self.timer("rhs"):
                b = disc.computeRhs()
                u0 = disc.initsolution(b)
            with self.timer("matrix"):
                A = disc.computeMatrix()
                self.As.append(A)
            with self.timer("linear_solver"):
                self.B.update(A=A)
                x = self.B.solve(b=b, x0=u0)
                u = b.from_flat_like(x) if isinstance(x, np.ndarray) else x

                res = np.linalg.norm(A @ u.flatten() - b.flatten())
            print(
                f"{ell:2d} "
                f"N={A.shape[0]:7d} "
                f"niter={self.B.niter:2d} "
                f"res={res:.3e}"
            )
            with self.timer("postproc"):
                postproc = disc.postProcess(u)
                if theta <= 1.0:
                    est = disc.computeEstimator(u)
                    postproc.setdefault("cell", {})
                    postproc["scalar"]["eta"] = est.eta
                    postproc["cell"]["eta"] = est.eta_cell

                # self.save(u=u)
            result = SimpleNamespace(u=u, postproc=postproc)
            for k, v in result.postproc['scalar'].items():
                print(f"{k:20s} : {v}")
            if plot_solution:
                with self.timer("plot"):
                    fig = plt.figure(figsize=(10, 8))
                    fig.suptitle(f"{self.application.__class__.__name__} nn={disc.mesh.nnodes:7d} ({ell=} )",
                                 fontsize=16)
                    outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
                    eta_plot = np.sqrt(result.postproc["cell"]["eta"])
                    disc.mesh.plot_boundary(fig=fig, outer=outer[0])
                    data = disc.plot_data(result.u, eta=eta_plot)
                    disc.mesh.plot(data=data, alpha=0.5, fig=fig, outer=outer[1])
                    plt.show()
            with self.timer("marking"):
                if theta > 1:
                    marked = np.ones(disc.mesh.ncells, dtype=bool)
                else:
                    eta = result.postproc["cell"]["eta"]
                    marked = marking.dorfler_marking(eta, theta=theta)
            with self.timer("refine"):
                mesh2, info = self.mesh_hierarchy.refine_nvb(marked, timer=mesh_timer, debug=False)

            with self.timer("create_discretization"):
                disc2 = self.discretize(mesh2)
                self.discs.append(disc2)

            with self.timer("interpolate"):
                transfer = disc.build_transfer_to_refined_mesh(info, disc_fine=disc2)
                self.transfers.append(transfer)
                u2 = transfer.interpolate(result.u)
                disc2.u0 = u2


            if plot_interpolation:
                with self.timer("plot_interpolation"):
                    fig = plt.figure(figsize=(10, 8))
                    fig.suptitle("Interpolation after NVB refinement", fontsize=16)
                    outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
                    data = disc.plot_data(result.u)
                    data2 = disc2.plot_data(u2)
                    disc.mesh.plot(data=data, fig=fig, outer=outer[0], alpha=0.1)
                    mesh2.plot(data=data2, fig=fig, outer=outer[1])
                    disc.mesh.plot(data=data, fig=fig, outer=outer[0], alpha=0.1)
                    mesh2.plot(data=data2, fig=fig, outer=outer[1])
                    plt.show()

        return result

    def discretize(self, mesh):
        return self.discretization_cls(
            mesh=mesh,
            application=self.application,
            disc_params=self.disc_params.copy(),
            problemdata=self.problemdata,
            timer=self.timer,
            verbose=self.verbose,
            **self.discretization_params,
        )

    def initial_guess(self):
        disc = self.discs[-1]
        b = disc.computeRhs()
        return disc.initsolution(b)

    def add_update(self, x, alpha, p):
        return x.from_flat_like(x.flatten() + alpha * p.flatten())

    def evaluate(self, x):
        disc = self.discs[-1]
        r = disc.computeForm(x)
        resn = np.linalg.norm(r)

        est = disc.computeEstimator(x)
        merit = np.sqrt(est.eta ** 2 + resn ** 2)

        return SimpleNamespace(
            residual=r,
            meritvalue=merit,
            norm_X=np.linalg.norm(x.flatten()),
        )

    def compute_newton_step(self, x, state, info):
        disc = self.discs[-1]

        A = disc.computeJacobian(x)
        r = disc.computeResidual(x)

        self.As.append(A)
        self.B.update(A=A)

        p_flat = self.B.solve(
            b=-r,
            x0=np.zeros_like(r),
        )
        p = x.from_flat_like(p_flat)

        return SimpleNamespace(
            dx=p,
            dx_norm=np.linalg.norm(p_flat),
            x=x,
            success=True,
        )

