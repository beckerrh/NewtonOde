import numpy as np
from FEM2D.models.model import Model
from FEM2D import fems
from Utility.analyticalfunction import AnalyticalFunction
# ================================================================= #
class EllipticBase(Model):
    r"""
    Class for the elliptic equation
    $$
    -\div(A \nabla T) + b\cdot\nabla u + c u= f         domain
    A\nabla\cdot n + alpha T = g  bdry
    $$
    After initialization, the function setMesh(mesh) has to be called
    Then, solve() solves the stationary problem
    Parameters in the constructor:
        fems: only p1, cr1, or rt0
        problemdata
        method
        masslumpedbdry, masslumpedvol
    Paramaters used from problemdata:
        kheat
        reaction
        alpha
        they can either be given as global constant, cell-wise constants, or global function
        - global constant is taken from problemdata.paramglobal
        - cell-wise constants are taken from problemdata.paramcells
        - problemdata.paramglobal is taken from problemdata.datafct and are called with arguments (color, xc, yc, zc)
    Possible parameters for computaion of postprocess:
        errors
        bdry_mean: computes mean temperature over boundary parts according to given color
        bdry_nflux: computes mean normal flux over boundary parts according to given color
    """
    def __format__(self, spec):
        if spec=='-':
            repr = super().__format__(spec)
            repr += f"\nfems={self.fem}"
            return repr
        return self.__repr__()
    def __repr__(self):
        repr = super().__repr__()
        repr += f"\nfems={self.fem}"
        return repr
    def __init__(self, **kwargs):
        # print(f"{kwargs=}")
        # print(f"{kwargs.keys()=}")
        self.fem = kwargs.pop('fem','cr1')
        self.linearsolver = kwargs.pop('linearsolver', 'pyamg')
        super().__init__(**kwargs)
    def createFem(self):
        self.hasconvection = 'convection' in self.disc_params \
                          or 'convection' in self.problemdata.params.data.keys()\
                          or 'convection' in self.problemdata.params.fct_glob.keys()
        if self.hasconvection:
            # print(f"{self.disc_params=}")
            self.convectionmethod = self.disc_params.pop('convmethod', 'lps')
            if self.convectionmethod == 'lps':
                self.lpsparam = self.disc_params.pop('lpsparam', 0.2)
        self.dirichletmethod = self.disc_params.pop('dirichletmethod','nitsche')
        if self.dirichletmethod=='nitsche':
            self.nitscheparam = self.disc_params.pop('nitscheparam', 10)
        if self.fem == 'p1': self.fem = fems.p1.P1()
        elif self.fem == 'cr1': self.fem = fems.cr1.CR1()
        else:
            self.rt = fems.rt0.RT0()
            self.d0 = fems.d0.D0()
            self.fem = "RT0-D0"
    def meshSet(self):
        if hasattr(self, 'A'):
            del self.A
        self._checkProblemData()
        self.kheatcell = self.compute_cell_vector_from_params('kheat', self.problemdata.params)
        if self.hasconvection:
            self.convdata = fems.data.ConvectionData()
            rt = fems.rt0.RT0(mesh=self.mesh)
            if 'convection' in self.problemdata.params.fct_glob:
                convection_given = self.problemdata.params.fct_glob['convection']
                if not isinstance(convection_given, list):
                    p = "problemdata.params.fct_glob['convection']"
                    raise ValueError(f"need '{p}' as a list of length dim of str or AnalyticalSolution")
                elif isinstance(convection_given[0],str):
                    self.convection_fct = [AnalyticalFunction(expr=e) for e in convection_given]
                else:
                    self.convection_fct = convection_given
                    if not isinstance(convection_given[0], AnalyticalFunction):
                        raise ValueError(f"convection should be given as 'str' and not '{type(convection_given[0])}'")
                if len(self.convection_fct) != self.mesh.dimension:
                    raise ValueError(f"{self.mesh.dimension=} {self.problemdata.params.fct_glob['convection']=}")
                # print(f"{convection_given=}")
                self.convdata.betart = rt.interpolate(self.convection_fct)
            else:
                data, fem, stack_storage = self.problemdata.params.data['convection']
                self.convdata.betart = rt.interpolateFromFem(data, fem, stack_storage)
            self.convdata.betacell = rt.toCell(self.convdata.betart)
            colorsinflow = self.findInflowColors()
            colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
            # print("betart shape", self.convdata.betart.shape)
            # print("nfaces", self.mesh.nfaces)
            # print("bdrylabels", {c: faces.tolist() for c, faces in self.mesh.bdrylabels.items()})
            # print("colorsinflow", colorsinflow)
            # print("colorsdir", colorsdir)
            if not set(colorsinflow).issubset(set(colorsdir)):
                raise ValueError(f"Inflow boundaries need to be subset of Dirichlet boundaries {colorsinflow=} {colorsdir=}")
    def findInflowColors(self):
        colors=[]
        for color in self.mesh.bdrylabels.keys():
            faces = self.mesh.bdrylabels[color]
            if np.any(self.convdata.betart[faces]<-1e-10): colors.append(color)
        return colors
    def _checkProblemData(self):
        if self.verbose: print(f"checking problem data {self.problemdata=}")
        bdrycond = self.problemdata.bdrycond
        for color in self.mesh.bdrylabels:
            if not color in bdrycond.type: raise ValueError(f"color={color} not in bdrycond={bdrycond}")
            if bdrycond.type[color] in ["Robin"]:
                if not color in bdrycond.param:
                    raise ValueError(f"Robin condition needs paral 'alpha' color={color} bdrycond={bdrycond}")
            if bdrycond.type[color] == "Dirichlet":
                if not color in bdrycond.fct:
                    bdrycond.fct[color] = lambda x,y,z: 0
                # raise ValueError(f"Dirichlet condition needs fct for color={color} bdrycond={bdrycond}")
    def defineRhsAnalyticalSolution(self, solexact_list):
        solexact = solexact_list[0]
        def _fctu(x, y, z):
            kheat = self.problemdata.params.scal_glob['kheat']
            beta = self.convection_fct
            rhs = np.zeros(x.shape)
            for i in range(self.mesh.dimension):
                rhs += beta[i](x,y,z) * solexact.d(i, x, y, z)
                rhs -= kheat * solexact.dd(i, i, x, y, z)
            return rhs
        def _fctu2(x, y, z):
            kheat = self.problemdata.params.scal_glob['kheat']
            rhs = np.zeros(x.shape)
            for i in range(self.mesh.dimension):
                rhs -= kheat * solexact.dd(i, i, x, y, z)
            return rhs
        if self.hasconvection: return _fctu
        return _fctu2
    def defineNeumannAnalyticalSolution(self, problemdata, color, solexact):
        solexact = solexact[0]
        # solexact = problemdata.solexact
        def _fctneumann(x, y, z, nx, ny, nz):
            kheat = self.problemdata.params.scal_glob['kheat']
            rhs = np.zeros(x.shape)
            normals = nx, ny, nz
            for i in range(self.mesh.dimension):
                rhs += kheat * solexact.d(i, x, y, z) * normals[i]
            return rhs
        return _fctneumann
    def defineRobinAnalyticalSolution(self, problemdata, color, solexact):
        solexact = solexact[0]
        # solexact = problemdata.solexact
        alpha = problemdata.bdrycond.param[color]
        kheat = self.problemdata.params.scal_glob['kheat']
        def _fctrobin(x, y, z, nx, ny, nz):
            rhs = np.zeros(x.shape)
            normals = nx, ny, nz
            rhs += alpha*solexact(x, y, z)
            for i in range(self.mesh.dimension):
                rhs += kheat * solexact.d(i, x, y, z) * normals[i]
            return rhs
        return _fctrobin
    def setParameter(self, paramname, param):
        if paramname == "dirichlet_strong": self.fem.dirichlet_strong = param
        else:
            if not hasattr(self, self.paramname):
                raise NotImplementedError("{} has no paramater '{}'".format(self, self.paramname))
            cmd = "self.{} = {}".format(self.paramname, param)
            eval(cmd)
