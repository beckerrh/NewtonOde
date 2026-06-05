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
class Model(object):
    def __format__(self, spec):
        if spec=='-':
            repr = f"fem={self.fem}"
            return repr
        return self.__repr__()
    def __repr__(self):
        if hasattr(self, 'mesh'):
            repr = f"mesh={self.mesh}"
        else:
            repr = "no mesh\n"
        repr += f"problemdata={self.problemdata}"
        repr += f"\ndisc_params={self.disc_params}"
        repr += f"\n{self.timer}"
        return repr
    def __init__(self, **kwargs):
        # print(f"Model {kwargs=}")
        self.stack_storage = kwargs.pop("stack_storage", False)
        self.verbose = kwargs.pop('verbose', 0)
        self.timer = Utility.timer.Timer()
        self.application = kwargs.pop('application', None)
        if self.application is None:
            raise ValueError(f"Model needs application (since 22/04/23)")
        self.problemdata = self.application.problemdata
        self.disc_params = kwargs.pop('disc_params', {})
        self.linear_solver = kwargs.pop("linear_solver", "spsolve")
        self.linear_solver_params = kwargs.pop("linear_solver_params", {})
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
        if len(kwargs.keys()):
            raise ValueError(f"*** unused arguments {kwargs=}")

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
    def afem_loop(self, niter, theta=0.9, plotting=False, mesh_timer=None):
        results = []

        for ell in range(niter):
            result = self.afem_step(
                ell,
                theta=theta,
                plotting=plotting,
                mesh_timer=mesh_timer,
            )
            results.append(result)

        return results

    def afem_step(self, ell, theta=0.9, plotting=False, mesh_timer=None):
        if plotting:
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
                u = self.B.solve(b=b, x0=u0)
            res = np.linalg.norm(A @ u - b)
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
            if plotting:
                with self.timer("plot"):
                    fig = plt.figure(figsize=(10, 8))
                    fig.suptitle(f"{self.application.__class__.__name__} nn={disc.mesh.nnodes:7d} ({ell=} )",
                                 fontsize=16)
                    outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
                    disc.mesh.plot_boundary(fig=fig, outer=outer[0])
                    eta_plot = np.sqrt(result.postproc["cell"]["eta"])
                    if result.u.shape[0] == disc.mesh.nnodes:
                        data = {"point": {"u": result.u}, "cell": {'k': disc.kheatcell, 'eta': eta_plot}, "global": {}}
                    else:
                        data = {"point": {"u": disc.fem.to_p1(result.u)}, "cell": {'k': disc.kheatcell, 'eta': eta_plot}, "global": {}}
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
                # mesh2, info = heat.mesh.refine_nvb(marked, timer=mesh_timer, debug=False)
            # in Model.afem_step


            with self.timer("create_discretization"):
                disc2 = self.discretize(mesh2)
                self.discs.append(disc2)

            with self.timer("interpolate"):
                transfer = disc.build_transfer_to_refined_mesh(info, disc_fine=disc2)
                self.transfers.append(transfer)
                u2 = transfer.interpolate(result.u)
                disc2.u0 = u2


            if plotting:
                with self.timer("plot_interpolation"):
                    fig = plt.figure(figsize=(10, 8))
                    fig.suptitle("Interpolation after NVB refinement", fontsize=16)
                    outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
                    if result.u.shape[0] == disc.mesh.nnodes:
                        data = {"point": {"u": result.u}, "cell": {}, "global": {}}
                        data2 = {"point": {"u": u2}, "cell": {}, "global": {}}
                    else:
                        data = {"point": {"u": disc.fem.to_p1(result.u)}, "cell": {}, "global": {}}
                        data2 = {"point": {"u": disc2.fem.to_p1(u2)}, "cell": {}, "global": {}}
                    disc.mesh.plot(data=data, fig=fig, outer=outer[0], alpha=0.1)
                    mesh2.plot(data=data2, fig=fig, outer=outer[1])
                    plt.show()

        return result

# ------------------------------------- #
if __name__ == '__main__':
    raise ValueError("unit tests to be written")
