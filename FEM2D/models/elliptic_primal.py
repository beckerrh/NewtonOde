import numpy as np
from FEM2D.models.elliptic_base import EllipticBase

# ================================================================= #
class EllipticPrimal(EllipticBase):
    def __init__(self, **kwargs):
        # print(f"{kwargs.keys()=}")
        super().__init__(**kwargs)
    def meshSet(self):
        super().meshSet()
        self.fem.setMesh(self.mesh)
        colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsflux = self.problemdata.postproc.colorsOfType("bdry_nflux")
        if self.dirichletmethod!="nitsche":
            self.bdrydata = self.fem.prepareBoundary(colorsdirichlet, colorsflux)
    def getNcomps(self, mesh):
        return [1]
    def getSystemSize(self):
        ns = [self.fem.nunknowns()]
        return ns
    def computeMassMatrix(self):
        lumped = self.disc_params.get('masslumped', False)
        return self.fem.computeMassMatrix(lumped=lumped)
    def computeForm(self, u, coeffmass=None):
        if not hasattr(self, 'A'):
            self.A = self.computeMatrix()
        # du2 = self.A@u
        du = np.zeros_like(u)
        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")
        self.fem.computeFormDiffusion(du, u, self.kheatcell)
        if self.hasconvection:
            self.fem.computeFormTransportCellWise(du, u, self.convdata, type='centered')
            if hasattr(self.fem, "computeFormJump"):
                self.fem.computeFormJump(du, u, self.convdata.betart)
            if self.convectionmethod == 'lps':
                self.fem.computeFormLps(du, u, self.convdata.betart, lpsparam=self.lpsparam)
        if coeffmass is not None:
            self.fem.massDot(du, u, coeff=coeffmass)
        self.fem.massDotBoundary(du, u, colorsrobin, bdrycond.param, lumped=True)
        if self.dirichletmethod!="nitsche":
            self.fem.vectorBoundaryStrongEqual(du, u, self.bdrydata)
        else:
            self.fem.computeFormNitscheDiffusion(self.nitscheparam, du, u, self.kheatcell, colorsdir)
        # if not np.allclose(du,du2):
        #     # f = (f"\n{du[self.bdrydata.facesdirall]}\n{du2[self.bdrydata.facesdirall]}")
        #     raise ValueError(f"{np.linalg.norm(du-du2)}\n{du=}\n{du2=}")
        return du
    def computeMatrix(self, u=None, coeffmass=None):
        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")
        A = self.fem.computeMatrixDiffusion(self.kheatcell)
        A += self.fem.computeBdryMassMatrix(colorsrobin, bdrycond.param, lumped=True)
        if self.hasconvection:
            A += self.fem.computeMatrixTransportCellWise(self.convdata, type='centered')
            if hasattr(self.fem, 'computeMatrixJump'):
                A += self.fem.computeMatrixJump(self.convdata.betart)
            if self.convectionmethod == 'lps':
                A += self.fem.computeMatrixLps(self.convdata.betart, lpsparam=self.lpsparam)
        if coeffmass is not None:
            A += self.fem.computeMassMatrix(coeff=coeffmass)
        if self.dirichletmethod!="nitsche":
            A = self.fem.matrixBoundaryStrong(A, self.bdrydata)
        else:
            A += self.fem.computeMatrixNitscheDiffusion(self.nitscheparam, diffcoff=self.kheatcell, colors=colorsdir)
        return A
    def computeRhs(self, b=None, coeffmass=None, u=None):
        if b is None:
            b = np.zeros(self.fem.nunknowns())
        else:
            if b.shape[0] != self.fem.nunknowns(): raise ValueError(f"{b.shape=} {self.fem.nunknowns()=}")
        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")
        colorsneu = bdrycond.colorsOfType("Neumann")
        if 'rhs' in self.problemdata.params.fct_glob:
            fp1 = self.fem.interpolate(self.problemdata.params.fct_glob['rhs'])
            self.fem.massDot(b, fp1)
            # if hasattr(self, 'convdata'): self.fems.massDotSupg(b, fp1, self.convdata)
        if 'rhscell' in self.problemdata.params.fct_glob:
            fp1 = self.fem.interpolateCell(self.problemdata.params.fct_glob['rhscell'])
            self.fem.massDotCell(b, fp1)
        if 'rhspoint' in self.problemdata.params.fct_glob:
            self.fem.computeRhsPoint(b, self.problemdata.params.fct_glob['rhspoint'])
        if self.dirichletmethod=="nitsche":
            self.fem.computeRhsNitscheDiffusion(self.nitscheparam, b, self.kheatcell, colorsdir, udir=None, bdrycondfct=bdrycond.fct)
        else:
            self.fem.vectorBoundaryStrong(b, bdrycond, self.bdrydata)
        if self.hasconvection:
            # fp1 = self.fems.interpolateBoundary(self.mesh.bdrylabels.keys(), bdrycond.fct)
            fp1 = self.fem.interpolateBoundary(colorsdir, bdrycond.fct)
            self.fem.massDotBoundary(b, fp1, coeff=-np.minimum(self.convdata.betart, 0))
        #Fourier-Robin
        fp1 = self.fem.interpolateBoundary(colorsrobin, bdrycond.fct, lumped=True)
        # self.fems.massDotBoundary(b, fp1, colors=colorsrobin, lumped=True, coeff=bdrycond.param)
        self.fem.massDotBoundary(b, fp1, colors=colorsrobin, lumped=True, coeff=1)
        #Neumann
        fp1 = self.fem.interpolateBoundary(colorsneu, bdrycond.fct)
        self.fem.massDotBoundary(b, fp1, colorsneu)
        if coeffmass is not None:
            assert u is not None
            self.fem.massDot(b, u, coeff=coeffmass)
        if hasattr(self, 'bdrydata'):
            self.fem.vectorBoundaryStrong(b, bdrycond, self.bdrydata)
        return b
    def postProcess(self, u):
        data = {'scalar':{}}
        if self.application.exactsolution:
            solexact = self.application.exactsolution[0]
            data['scalar']['err_L2c'], ec = self.fem.computeErrorL2Cell(solexact, u)
            data['scalar']['err_L2n'], en = self.fem.computeErrorL2 (solexact, u)
            data['scalar']['err_H1'] = self.fem.computeErrorFluxL2  (solexact, u)
            data['scalar']['err_Flux'] = self.fem.computeErrorFluxL2(solexact, u, self.kheatcell)
            data['cell'] = {}
            data['cell']['err'] = ec
        if self.problemdata.postproc:
            types = ["bdry_mean", "bdry_fct", "bdry_nflux", "pointvalues", "meanvalues", "linemeans"]
            for name, type in self.problemdata.postproc.type.items():
                colors = self.problemdata.postproc.colors(name)
                if type == types[0]:
                    data['scalar'][name] = self.fem.computeBdryMean(u, colors)
                elif type == types[1]:
                    data['scalar'][name] = self.fem.computeBdryFct(u, colors)
                elif type == types[2]:
                    if self.dirichletmethod == 'nitsche':
                        udir = self.fem.interpolateBoundary(colors, self.problemdata.bdrycond.fct)
                        data['scalar'][name] = self.fem.computeBdryNormalFluxNitsche(self.nitscheparam, u, colors, udir, self.kheatcell)
                    else:
                        data['scalar'][name] = self.fem.computeBdryNormalFlux(u, colors, self.bdrydata, self.problemdata.bdrycond, self.kheatcell)
                elif type == types[3]:
                    data['scalar'][name] = self.fem.computePointValues(u, colors)
                elif type == types[4]:
                    data['scalar'][name] = self.fem.computeMeanValues(u, colors)
                elif type == types[5]:
                    data['scalar'][name] = self.fem.computeLineValues(u, colors)
                else:
                    raise ValueError(f"unknown postprocess type '{type}' for key '{name}'\nknown types={types=}")
        if hasattr(self.fem, "computeEstimatorJumpP1"):
            if "rhs" in self.problemdata.params.fct_glob:
                xc, yc, zc = self.mesh.cell_centers.T
                rhs_cell = self.problemdata.params.fct_glob["rhs"](xc, yc, zc)
            else:
                rhs_cell = np.zeros(self.mesh.ncells)  # for first jump-only test

            eta, eta2 = self.fem.computeEstimatorJumpP1(
                u,
                rhs_cell=rhs_cell,
                diffcell=self.kheatcell,
            )
            data["scalar"]["eta"] = eta
            data.setdefault("cell", {})
            data["cell"]["eta"] = np.sqrt(eta2)
        return data
    def pyamg_solver_args(self, args):
        if self.hasconvection:
            args['symmetric'] = False
            args['smoother'] = 'schwarz'
        else:
            args['symmetric'] = True
