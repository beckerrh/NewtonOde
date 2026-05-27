# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import shutil, pathlib
import numpy as np
from types import SimpleNamespace

import Utility.timer
import FEM2D.models.problemdata
from FEM2D.linalg import vectorview

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
    # def setMesh(self, mesh):
    #     self.mesh = mesh
    #     self.problemdata.check(self.mesh)
    #     if self.verbose: print(f"{self.mesh=}")
    #     for name in ("LS", "A", "bdrydata", "convdata"):
    #         if hasattr(self, name):
    #             delattr(self, name)
    #     self.ncomps = self.getNcomps(self.mesh)
    #     # self.meshSet()
    #     # ns = self.getSystemSize()
    #     # self.vectorview = vectorview.VectorView(ncomps=self.ncomps, ns=ns, stack_storage=self.stack_storage)
    #     if not hasattr(self, "exactsolution_created"):
    #         if self.application.exactsolution is not None:
    #             self.exactsolution_created=True
    #             self.application.createExactSolution(self.mesh, self.ncomps)
    #         if self.application.generatePDforES:
    #             self.generatePoblemDataForAnalyticalSolution()
    def getNcomps(self, mesh):
        return [1]
    def getSystemSize(self):
        return [self.fem.nunknowns()]
    def createFem(self):
        raise NotImplementedError(f"createFem has to be overwritten")
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
        print(f"{self.application.exactsolution=} {self.mesh.labels.boundary=}")
        solexact = self.application.exactsolution
        self.problemdata.params.fct_glob['rhs'] = self.defineRhsAnalyticalSolution(solexact)
        if hasattr(self, 'time'):
            self.problemdata.params.fct_glob['initial_condition'] = self.defineInitialConditionAnalyticalSolution(solexact)
        for color in self.mesh.labels.boundary:
            cmd = f"self.define{bdrycond.type[color]}AnalyticalSolution(self.problemdata,{color},solexact)"
            # print(f"cmd={cmd}")
            bdrycond.fct[color] = eval(cmd)
    def initsolution(self, b):
        if isinstance(b,tuple):
            # raise KeyError("i don't know how to handle {type(b)=}")
            return [np.copy(bi) for bi in b]
        return b.copy()
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

# ------------------------------------- #
if __name__ == '__main__':
    raise ValueError("unit tests to be written")
