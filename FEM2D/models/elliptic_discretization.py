import numpy as np
from FEM2D import fems
from Utility.analyticalfunction import AnalyticalFunction

# ================================================================= #
class EllipticDiscretization:
    def __init__(self, mesh, application, fem_name, disc_params, problemdata, **kwargs):
        self.verbose = kwargs.get('verbose', False)
        self.mesh = mesh
        self.application = application
        self.disc_params = disc_params
        self.problemdata = problemdata
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
        if fem_name == 'p1': self.fem = fems.p1.P1()
        elif fem_name == 'cr1': self.fem = fems.cr1.CR1()
        else: raise NotImplementedError(f"{self.fem=}")
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
            # print("bdrylabels", {c: faces.tolist() for c, faces in self.mesh.labels.boundary.items()})
            # print("colorsinflow", colorsinflow)
            # print("colorsdir", colorsdir)
            if not set(colorsinflow).issubset(set(colorsdir)):
                raise ValueError(f"Inflow boundaries need to be subset of Dirichlet boundaries {colorsinflow=} {colorsdir=}")
        self.fem.setMesh(self.mesh)
        colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsflux = self.problemdata.postproc.colorsOfType("bdry_nflux")
        if self.dirichletmethod != "nitsche":
            self.bdrydata = self.fem.prepareBoundary(colorsdirichlet, colorsflux)
    def findInflowColors(self):
        colors=[]
        for color in self.mesh.labels.boundary.keys():
            faces = self.mesh.labels.boundary[color]
            if np.any(self.convdata.betart[faces]<-1e-10): colors.append(color)
        return colors
    def initsolution(self, b):
        if isinstance(b,tuple):
            # raise KeyError("i don't know how to handle {type(b)=}")
            return [np.copy(bi) for bi in b]
        return b.copy()
    def compute_cell_vector_from_params(self, name, params):
        if name in params.fct_glob:
            fct = np.vectorize(params.fct_glob[name])
            arr = np.empty(self.mesh.ncells)
            for color, cells in self.mesh.labels.cell.items():
                xc, yc, zc = self.mesh.geometry.cell_centers[cells].T
                arr[cells] = fct(color, xc, yc, zc)
        elif name in params.scal_glob:
            arr = np.full(self.mesh.ncells, params.scal_glob[name])
        elif name in params.scal_cells:
            arr = np.empty(self.mesh.ncells)
            for color in params.scal_cells[name]:
                arr[self.mesh.labels.cell[color]] = params.scal_cells[name][color]
        else:
            msg = f"{name} should be given in 'fct_glob' or 'scal_glob' or 'scal_cells' (problemdata.params)"
            raise ValueError(msg)
        return arr
    def _checkProblemData(self):
        if self.verbose: print(f"checking problem data {self.problemdata=}")
        bdrycond = self.problemdata.bdrycond
        for color in self.mesh.labels.boundary:
            if not color in bdrycond.type: raise ValueError(f"color={color} not in bdrycond={bdrycond}")
            if bdrycond.type[color] in ["Robin"]:
                if not color in bdrycond.param:
                    raise ValueError(f"Robin condition needs paral 'alpha' color={color} bdrycond={bdrycond}")
            if bdrycond.type[color] == "Dirichlet":
                if not color in bdrycond.fct:
                    bdrycond.fct[color] = lambda x,y,z: 0
                # raise ValueError(f"Dirichlet condition needs fct for color={color} bdrycond={bdrycond}")
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
            # fp1 = self.fems.interpolateBoundary(self.mesh.labels.boundary.keys(), bdrycond.fct)
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
                xc, yc, zc = self.mesh.geometry.cell_centers.T
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
