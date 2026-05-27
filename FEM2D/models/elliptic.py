import numpy as np
from FEM2D.models.model import Model
from FEM2D.models.elliptic_discretization import EllipticDiscretization

# ================================================================= #
class Elliptic(Model):
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
    def __init__(self, **kwargs):
        self.fem_name = kwargs.pop('fem','cr1')
        super().__init__(**kwargs)
    def discretize(self, mesh):
        return EllipticDiscretization(
            mesh=mesh,
            application=self.application,
            fem_name=self.fem_name,
            disc_params=self.disc_params,
            problemdata=self.problemdata,
        )


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

    def getNcomps(self, mesh):
        return [1]
    def getSystemSize(self):
        ns = [self.fem.nunknowns()]
        return ns
    def pyamg_solver_args(self, args):
        if self.hasconvection:
            args['symmetric'] = False
            args['smoother'] = 'schwarz'
        else:
            args['symmetric'] = True
