# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import shutil, pathlib
import numpy as np

import Utility.timer
import FEM2D.models.problemdata
from FEM2D.linalg import vectorview
from FEM2D import fems

#=================================================================#
class Model(object):
    def __format__(self, spec):
        if spec=='-':
            repr = f"fem={self.fem}"
            repr += f"\tlinearsolver={self.linearsolver}"
            return repr
        return self.__repr__()
    def __repr__(self):
        if hasattr(self, 'mesh'):
            repr = f"mesh={self.mesh}"
        else:
            repr = "no mesh\n"
        repr += f"problemdata={self.problemdata}"
        repr += f"\nlinearsolver={self.linearsolver}"
        repr += f"\ndisc_params={self.disc_params}"
        repr += f"\n{self.timer}"
        return repr
    def __init__(self, **kwargs):
        # print(f"Model {kwargs=}")
        self.stack_storage = kwargs.pop("stack_storage", False)
        self.mode = kwargs.pop('mode', 'linear')
        self.verbose = kwargs.pop('verbose', 0)
        self.timer = Utility.timer.Timer()
        self.application = kwargs.pop('application', None)
        if self.application is None:
            raise ValueError(f"Model needs application (since 22/04/23)")
        self.problemdata = self.application.problemdata
        if not hasattr(self,'linearsolver'):
            self.linearsolver = kwargs.pop('linearsolver', 'spsolve')
        self.disc_params = kwargs.pop('disc_params', {})
        if not hasattr(self,'scale_ls'):
            self.scale_ls = kwargs.pop('scale_ls', False)
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
        if 'mesh' in kwargs.keys():
            self.mesh = kwargs.pop('mesh')
        else:
            self.mesh =self.application.createMesh()
        # print(f"{kwargs.keys()=}")
        if len(kwargs.keys()):
            raise ValueError(f"*** unused arguments {kwargs=}")
        self.createFem()
        self.setMesh(self.mesh)
    def setMesh(self, mesh):
        self.mesh = mesh
        self.problemdata.check(self.mesh)
        if self.verbose: print(f"{self.mesh=}")
        for name in ("LS", "A", "bdrydata", "convdata"):
            if hasattr(self, name):
                delattr(self, name)
        self.ncomps = self.getNcomps(self.mesh)
        self.meshSet()
        ns = self.getSystemSize()
        self.vectorview = vectorview.VectorView(ncomps=self.ncomps, ns=ns, stack_storage=self.stack_storage)
        if not hasattr(self, "exactsolution_created"):
            if self.application.exactsolution is not None:
                self.exactsolution_created=True
                self.application.createExactSolution(self.mesh, self.ncomps)
            if self.application.generatePDforES:
                self.generatePoblemDataForAnalyticalSolution()
    def getNcomps(self, mesh):
        return [1]
    def getSystemSize(self):
        return [self.fem.nunknowns()]
    def tofemvector(self, u):
        return fems.femvector.FemVector(data = u, vectorview=self.vectorview, fems=[self.fem])
    def createFem(self):
        raise NotImplementedError(f"createFem has to be overwritten")
    def solve(self):
        if self.mode=='dynamic':
            return self.dynamic()
        return self.static(method=self.mode)
    def defineDirichletAnalyticalSolution(self, problemdata, color, solexact):
        ncomp = self.ncomps[0]
        if ncomp==1:
            return solexact[0]
        else:
            from functools import partial
            solexact = self.application.exactsolution
            def _solexactdir(x, y, z, icomp):
                return solexact[icomp](x, y, z)
            return [partial(_solexactdir, icomp=icomp) for icomp in range(ncomp)]
    def generatePoblemDataForAnalyticalSolution(self):
        bdrycond = self.problemdata.bdrycond
        print(f"{self.application.exactsolution=} {self.mesh.bdrylabels=}")
        solexact = self.application.exactsolution
        self.problemdata.params.fct_glob['rhs'] = self.defineRhsAnalyticalSolution(solexact)
        if hasattr(self, 'time'):
            self.problemdata.params.fct_glob['initial_condition'] = self.defineInitialConditionAnalyticalSolution(solexact)
        for color in self.mesh.bdrylabels:
            cmd = f"self.define{bdrycond.type[color]}AnalyticalSolution(self.problemdata,{color},solexact)"
            # print(f"cmd={cmd}")
            bdrycond.fct[color] = eval(cmd)
    def compute_cell_vector_from_params(self, name, params):
        if name in params.fct_glob:
            fct = np.vectorize(params.fct_glob[name])
            arr = np.empty(self.mesh.ncells)
            for color, cells in self.mesh.cellsoflabel.items():
                xc, yc, zc = self.mesh.cell_centers[cells].T
                arr[cells] = fct(color, xc, yc, zc)
        elif name in params.scal_glob:
            arr = np.full(self.mesh.ncells, params.scal_glob[name])
        elif name in params.scal_cells:
            arr = np.empty(self.mesh.ncells)
            for color in params.scal_cells[name]:
                arr[self.mesh.cellsoflabel[color]] = params.scal_cells[name][color]
        else:
            msg = f"{name} should be given in 'fct_glob' or 'scal_glob' or 'scal_cells' (problemdata.params)"
            raise ValueError(msg)
        return arr
    def initsolution(self, b):
        if isinstance(b,tuple):
            # raise KeyError("i don't know how to handle {type(b)=}")
            return [np.copy(bi) for bi in b]
        return b.copy()
    def computelinearSolver(self, A):
        # print(f"{self.linearsolver=} {self.scale_ls=}")
        if isinstance(self.linearsolver,str):
            args = {'method': self.linearsolver}
        else:
            args = self.linearsolver.copy()
        # args['matrix'] = A
        if args['method'] != 'spsolve':
            if self.scale_ls:
                if hasattr(A, 'scale_matrix'):
                    A.scale_matrix()
                args['scale'] = self.scale_ls
            args['matrix'] = A
            if args['method'] != 'pyamg':
                if hasattr(A,'matvec'):
                    # args['matvec'] = A.matvec
                    args['n'] = A.nall
                else:
                    # args['matvec'] = lambda x: np.matmul(A,x)
                    args['n'] = A.shape[0]
            else:
                self.pyamg_solver_args(args)
        # print(f"{args=}")
        return FEM2D.linalg.linalg.getLinearSolver(**args)
    def static(self, **kwargs):
        method = kwargs.pop('method','newton')
        u = kwargs.pop('u',None)
        result = FEM2D.models.problemdata.Results()
        self.b = kwargs.pop('b',self.computeRhs())
        if u is None:
            u = self.initsolution(self.b)
        if method == 'linear':
            try:
                if not hasattr(self,'A'):
                    with self.timer('computeMatrix'):
                        self.A = self.computeMatrix()
                with self.timer('computelinearSolver'):
                    self.LS = self.computelinearSolver(self.A)
                with self.timer('solve'):
                    u = self.LS.solve(A=self.A, b=self.b, x0=u)
                niterlin = self.LS.niter
            except Warning:
                raise ValueError(f"matrix is singular {self.A.shape=} {self.A.diagonal()=}")
            iter={'lin':niterlin}
        else:
            raise ValueError(f"method {method} not implemented")
        pp = self.postProcess(u)
        if hasattr(self.application, "changepostproc"):
            self.application.changepostproc(pp['scalar'])

        result.setData(pp, timer=self.timer, iter=iter)
        self.save(u=u)
        return result, self.tofemvector(u)
    # def newMatrix(self, u):
    #     if not hasattr(self, 'timeiter'):
    #         coeffmass = None
    #     else:
    #         coeffmass = self.coeffmass
    #     self.A = self.computeMatrix(u=u, coeffmass=coeffmass)
    #     if hasattr(self.A, 'scale_matrix') and self.scale_ls:
    #         self.A.scale_matrix()
    #     if hasattr(self, 'LS'):
    #         # self.LS = self.computelinearSolver(self.A)
    #         self.LS.update(self.A)
    #     else:
    #         self.LS = self.computelinearSolver(self.A)
    def newMatrix(self, u):
        coeffmass = None if not hasattr(self, 'timeiter') else self.coeffmass

        self.A = self.computeMatrix(u=u, coeffmass=coeffmass)

        if hasattr(self.A, 'scale_matrix') and self.scale_ls:
            self.A.scale_matrix()

        method = self.linearsolver if isinstance(self.linearsolver, str) else self.linearsolver.get("method")

        if method == "pyamg":
            self.LS = self.computelinearSolver(self.A)  # rebuild hierarchy
        elif hasattr(self, 'LS'):
            self.LS.update(self.A)
        else:
            self.LS = self.computelinearSolver(self.A)
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
    def load_data(self, iter=None, datadir=None, name= "sol", add=''):
        u = self.tofemvector(self.load(iter, datadir, name, add))
        return u.tovisudata()
    def get_t(self, datadir=None):
        if datadir is None: datadir=self.datadir
        times = np.load(self.datadir / "time.npy")
        return times
    def get_postprocs_dynamic(self):
        data = {'time': np.load(self.datadir/"time.npy"), 'postproc':{}}
        from pathlib import Path
        p = Path(self.datadir)
        for q in p.glob('postproc*.npy'):
            pname = '_'.join(str(q.parts[-1]).split('.')[0].split('_')[1:])
            # print(f"{pname=} {q=}")
            data['postproc'][pname] = np.load(q)
        return data
    def sol_to_vtu(self, **kwargs):
        # print(f"sol_to_vtu {kwargs=}")
        niter = kwargs.pop('niter', None)
        suffix = kwargs.pop('suffix', '')
        solnamebase = "sol" + suffix
        if niter is None:
            u = kwargs.pop('u', None)
            if u is None:
                filename = self.datadir / (solnamebase + ".npy")
                print(f"loading {filename=}")
                u = np.load(filename)
            data = self.sol_to_data(u)
            filename = self.datadir / (solnamebase + ".vtu")
            print(f"writing {filename=}")
            self.mesh.write(filename, data=data)
            return
        for iter in range(niter):
            solname = solnamebase + f"_{iter:05d}"
            filename = self.datadir / (solnamebase + ".npy")
            u = np.load(filename)
            data = self.sol_to_data(u)
            filename = self.datadir/(solname + ".vtu")
            self.mesh.write(filename, data=data)

# ------------------------------------- #
if __name__ == '__main__':
    raise ValueError("unit tests to be written")
