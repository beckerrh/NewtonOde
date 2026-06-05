import numpy as np
from types import SimpleNamespace

from Utility.analyticalfunction import AnalyticalFunction
from .discretization_base import DiscretizationBase
from ..fems import cr1, p1, rt0, mesh_transfer
from ..fems import data as femdata
from ..fems.diffusion import normalize_diffusion
from ..linalg.fem_vector import FemVector


# ================================================================= #
class EllipticDiscretization(DiscretizationBase):
    def __init__(self, mesh, application, fem_name, disc_params, problemdata, **kwargs):

        if fem_name == 'p1': fem = p1.P1()
        elif fem_name == 'cr1': fem = cr1.CR1()
        else: raise NotImplementedError(f"{fem_name=}")
        fem.setMesh(mesh)

        verbose = kwargs.pop("verbose", False)
        timer = kwargs.pop("timer", None)
        super().__init__(mesh, application, fems=[fem], block_names=["U"], verbose=verbose, timer=timer)


        self.problemdata = problemdata

        self.hasconvection = 'convection' in disc_params \
                          or 'convection' in self.problemdata.params.data.keys()\
                          or 'convection' in self.problemdata.params.fct_glob.keys()

        if application.exactsolution is not None:
            self.generatePoblemDataForAnalyticalSolution()
        if self.hasconvection:
            self.convectionmethod = disc_params.pop('convmethod', 'lps')
            if self.convectionmethod == 'lps':
                self.lpsparam = disc_params.pop('lpsparam', 0.2)

        self.dirichletmethod = disc_params.pop('dirichletmethod','nitsche')
        if self.dirichletmethod=='nitsche':
            self.nitscheparam = disc_params.pop('nitscheparam', 10)
            self.nitsche_lumped = disc_params.pop('nitsche_lumped', False)

        self._checkProblemData()
        self.kheatcell = self.compute_cell_vector_from_params('kheat', self.problemdata.params)
        self.diffkind, self.diffcell = normalize_diffusion(
            self.kheatcell,
            self.mesh.ncells,
            ncomp=self.application.ncomps[0],
            dim=self.mesh.dimension,
        )
        if self.hasconvection:
            self.convdata = femdata.ConvectionData()
            rt = rt0.RT0(mesh=self.mesh)
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
        colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsflux = self.problemdata.postproc.colorsOfType("bdry_nflux")
        if self.dirichletmethod != "nitsche":
            self.bdrydata = self.fem.prepareBoundary(colorsdirichlet, colorsflux)
        assert len(self.application.ncomps)==1
        if len(kwargs.keys()):
            raise ValueError(f"*** unused arguments {kwargs=}")
        if len(disc_params.keys()):
            raise ValueError(f"*** unused arguments {disc_params=}")


    # ================================================================= #
    def defineDirichletAnalyticalSolution(self, solexact_list):
        solexact = solexact_list[0]

        def _as_vector_value(vals, x):
            out = []
            for vi in vals:
                vi = np.asarray(vi)
                if vi.ndim == 0:
                    vi = np.full_like(x, float(vi), dtype=float)
                else:
                    vi = vi.astype(float)
                out.append(vi)
            return np.asarray(out)

        if isinstance(solexact, (list, tuple)):
            def _solexactdir(x, y, z, *args):
                z = np.zeros_like(x) if z is None else z
                vals = [ui(x, y, z) for ui in solexact]
                return _as_vector_value(vals, x)

            return _solexactdir

        def _solexactdir(x, y, z, *args):
            z = np.zeros_like(x) if z is None else z
            val = np.asarray(solexact(x, y, z))
            if val.ndim == 0:
                return np.full_like(x, float(val), dtype=float)
            return val.astype(float)

        return _solexactdir
    def defineRhsAnalyticalSolution(self, solexact_list):

        solexact = solexact_list[0]

        def _scalar_rhs(ui, x, y, z, with_convection):
            kheat = self.problemdata.params.scal_glob["kheat"]

            rhs = np.zeros(x.shape)

            for i in range(self.mesh.dimension):

                if with_convection:
                    beta = self.convection_fct
                    rhs += beta[i](x, y, z) * ui.d(i, x, y, z)

                rhs -= kheat * ui.dd(i, i, x, y, z)

            return rhs

        def _vector_rhs(x, y, z, with_convection):

            ncomp = self.application.ncomps[0]

            rhs = np.zeros((ncomp, x.size))

            for icomp in range(ncomp):
                rhs[icomp] = _scalar_rhs(
                    solexact[icomp],
                    x, y, z,
                    with_convection,
                )

            return rhs

        def _fctu(x, y, z):

            # vector-valued unknown
            if isinstance(solexact, (list, tuple)):
                return _vector_rhs(x, y, z, True)

            # scalar unknown
            return _scalar_rhs(solexact, x, y, z, True)

        def _fctu2(x, y, z):

            # vector-valued unknown
            if isinstance(solexact, (list, tuple)):
                return _vector_rhs(x, y, z, False)

            # scalar unknown
            return _scalar_rhs(solexact, x, y, z, False)

        if self.hasconvection:
            return _fctu

        return _fctu2
    def defineNeumannAnalyticalSolution(self, problemdata, color, solexact):
        solexact = solexact[0]

        def _fctneumann(x, y, z, nx, ny, nz):
            kheat = self.problemdata.params.scal_glob["kheat"]
            rhs = np.zeros(x.shape)
            normals = nx, ny, nz
            for i in range(self.mesh.dimension):
                rhs += kheat * solexact.d(i, x, y, z) * normals[i]
            return rhs

        return _fctneumann
    def defineRobinAnalyticalSolution(self, problemdata, color, solexact):
        solexact = solexact[0]
        alpha = problemdata.bdrycond.param[color]
        kheat = self.problemdata.params.scal_glob["kheat"]

        def _fctrobin(x, y, z, nx, ny, nz):
            rhs = np.zeros(x.shape)
            normals = nx, ny, nz
            rhs += alpha * solexact(x, y, z)
            for i in range(self.mesh.dimension):
                rhs += kheat * solexact.d(i, x, y, z) * normals[i]
            return rhs

        return _fctrobin

    def generatePoblemDataForAnalyticalSolution(self):

        bdrycond = self.problemdata.bdrycond
        solexact = self.application.exactsolution

        self.problemdata.params.fct_glob["rhs"] = \
            self.defineRhsAnalyticalSolution(solexact)

        if hasattr(self, "time"):
            self.problemdata.params.fct_glob["initial_condition"] = \
                self.defineInitialConditionAnalyticalSolution(solexact)

        for color in self.mesh.labels.boundary:

            bc_type = bdrycond.type[color]

            if bc_type == "Dirichlet":

                bdrycond.fct[color] = \
                    self.defineDirichletAnalyticalSolution(solexact)

            elif bc_type == "Neumann":

                bdrycond.fct[color] = \
                    self.defineNeumannAnalyticalSolution(
                        self.problemdata,
                        color,
                        solexact,
                    )

            elif bc_type == "Robin":

                bdrycond.fct[color] = \
                    self.defineRobinAnalyticalSolution(
                        self.problemdata,
                        color,
                        solexact,
                    )

            else:
                raise ValueError(f"unknown boundary type '{bc_type}'")
    # ================================================================= #


    def cell_coordinates_xyz(self):
        xc = self.mesh.geometry.cell_centers[:, 0]
        yc = self.mesh.geometry.cell_centers[:, 1]
        if self.mesh.dimension == 2:
            zc = np.zeros(self.mesh.ncells)
        else:
            zc = self.mesh.geometry.cell_centers[:, 2]
        return xc, yc, zc

    def findInflowColors(self):
        colors=[]
        for color in self.mesh.labels.boundary.keys():
            faces = self.mesh.labels.boundary[color]
            if np.any(self.convdata.betart[faces]<-1e-10): colors.append(color)
        return colors
    def initsolution(self, b):
        if getattr(self, "u0", None) is not None:
            u0 = self.u0
            self.u0 = None
            return u0.copy()
        return FemVector.zeros_like(b)
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
        self.fem.computeFormDiffusion(du, u, self.diffcell)
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
            self.fem.computeFormNitscheDiffusion(self.nitscheparam, du, u, self.diffcell, colorsdir, lumped=self.nitsche_lumped)
        # if not np.allclose(du,du2):
        #     # f = (f"\n{du[self.bdrydata.facesdirall]}\n{du2[self.bdrydata.facesdirall]}")
        #     raise ValueError(f"{np.linalg.norm(du-du2)}\n{du=}\n{du2=}")
        return du
    def computeMatrix(self, u=None, coeffmass=None):
        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")
        A = self.fem.computeMatrixDiffusion(self.diffcell)
        A += self.fem.computeBdryMassMatrix(colorsrobin, bdrycond.param, lumped=True)
        if self.hasconvection:
            A += self.fem.computeMatrixTransportCellWise(self.convdata, type='centered')
            if hasattr(self.fem, 'computeMatrixJump'):
                A += self.fem.computeMatrixJump(self.convdata.betart)
            if self.convectionmethod == 'lps':
                A += self.fem.computeMatrixLps(self.convdata.betart, lpsparam=self.lpsparam)
        if coeffmass is not None:
            A += self.fem.computeMassMatrix(coeff=coeffmass)
        if self.dirichletmethod != "nitsche":
            A = self.fem.matrixBoundaryStrong(A, self.bdrydata)
        else:
            A += self.fem.computeMatrixNitscheDiffusion(
                self.nitscheparam,
                diffcoff=self.diffcell,
                colors=colorsdir,
                lumped=self.nitsche_lumped,
            )

        ncomp = self.application.ncomps[0]
        if ncomp > 1:
            from scipy.sparse import block_diag
            A = block_diag([A] * ncomp, format="csr")

        return A

    def _component_boundary_function(self, f, icomp):
        ncomp = self.ncomps[0]

        def g(x, y, z, *args):
            x1 = np.atleast_1d(x)
            y1 = np.atleast_1d(y)
            z1 = np.atleast_1d(z)
            npts = x1.shape[0]

            try:
                val = f(x1, y1, z1, *args)
            except TypeError:
                val = f(x1, y1, z1)

            val = np.asarray(val, dtype=float)

            if val.shape == ():
                out = np.full(npts, float(val))
            elif val.shape == (npts,):
                if ncomp != 1:
                    raise ValueError(f"scalar boundary value for vector problem: {ncomp=}")
                out = val
            elif val.shape == (ncomp, npts):
                out = val[icomp]
            elif val.shape == (npts, ncomp):
                out = val[:, icomp]
            else:
                raise ValueError(
                    f"unexpected boundary value shape {val.shape=} "
                    f"expected (), {(npts,)}, {(ncomp, npts)} or {(npts, ncomp)}"
                )

            return out[0] if np.ndim(x) == 0 else out

        return g
    def computeRhs(self, b=None, coeffmass=None, u=None):
        ncomp = self.ncomps[0]
        ndof = self.fem.nunknowns()

        if b is None:
            b = self.newVector()
        elif not isinstance(b, FemVector):
            b = FemVector.from_flat(
                b,
                ns=self.ndofs,
                ncomps=self.ncomps,
                names=["u"],
            )

        B = b.block(0)

        if B.shape != (ncomp, ndof):
            raise ValueError(f"{B.shape=} expected {(ncomp, ndof)}")

        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")
        colorsneu = bdrycond.colorsOfType("Neumann")
        if "rhs" in self.problemdata.params.fct_glob:
            fp1 = np.asarray(self.fem.interpolate(self.problemdata.params.fct_glob["rhs"]))

            if fp1.ndim == 1:
                fp1 = fp1.reshape(1, -1)

            if fp1.shape != (ncomp, ndof):
                raise ValueError(f"{fp1.shape=} expected {(ncomp, ndof)}")

            for icomp in range(ncomp):
                self.fem.massDot(B[icomp], fp1[icomp])
        if "rhscell" in self.problemdata.params.fct_glob:
            fp1 = np.asarray(self.fem.interpolateCell(self.problemdata.params.fct_glob["rhscell"]))

            if fp1.ndim == 1:
                fp1 = fp1.reshape(1, -1)

            for icomp in range(ncomp):
                self.fem.massDotCell(B[icomp], fp1[icomp])
        if 'rhspoint' in self.problemdata.params.fct_glob:
            self.fem.computeRhsPoint(b, self.problemdata.params.fct_glob['rhspoint'])
        ncomp = self.application.ncomps[0]

        if self.dirichletmethod == "nitsche":
            for icomp in range(ncomp):
                bdrycondfct_i = {
                    color: self._component_boundary_function(bdrycond.fct[color], icomp)
                    for color in colorsdir
                }

                self.fem.computeRhsNitscheDiffusion(
                    self.nitscheparam,
                    B[icomp],
                    self.diffcell,
                    colorsdir,
                    udir=None,
                    bdrycondfct=bdrycondfct_i,
                    lumped=self.nitsche_lumped,
                )
        else:
            if ncomp != 1:
                raise NotImplementedError("strong Dirichlet for vector elliptic systems")
            self.fem.vectorBoundaryStrong(B[0], bdrycond, self.bdrydata)
        if self.hasconvection:
            fp1 = self.fem.interpolateBoundary(colorsdir, bdrycond.fct)

            for icomp in range(ncomp):
                self.fem.massDotBoundary(
                    B[icomp],
                    fp1 if ncomp == 1 else fp1[icomp],
                    coeff=-np.minimum(self.convdata.betart, 0),
                )
        #Fourier-Robin
        fp1 = self.fem.interpolateBoundary(colorsrobin, bdrycond.fct, lumped=True)
        # self.fems.massDotBoundary(b, fp1, colors=colorsrobin, lumped=True, coeff=bdrycond.param)
        for icomp in range(ncomp):
            self.fem.massDotBoundary(
                B[icomp],
                fp1 if ncomp == 1 else fp1[icomp],
                colors=colorsrobin,
                lumped=True,
                coeff=1,
            )
        #Neumann
        fp1 = self.fem.interpolateBoundary(colorsneu, bdrycond.fct)

        for icomp in range(ncomp):
            self.fem.massDotBoundary(
                B[icomp],
                fp1 if ncomp == 1 else fp1[icomp],
                colors=colorsneu,
            )
        if coeffmass is not None:
            assert u is not None
            self.fem.massDot(b, u, coeff=coeffmass)
        if hasattr(self, 'bdrydata'):
            self.fem.vectorBoundaryStrong(b, bdrycond, self.bdrydata)
        return b

    def postProcess(self, u):
        data = {'scalar':{}}
        ncomp = self.application.ncomps[0]

        if self.application.exactsolution:
            scal, cell = self.compute_errors_exact(u)
            data["scalar"].update(scal)
            data["cell"] = cell
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
                        data['scalar'][name] = self.fem.computeBdryNormalFluxNitsche(self.nitscheparam, u, colors, udir, self.diffcell)
                    else:
                        data['scalar'][name] = self.fem.computeBdryNormalFlux(u, colors, self.bdrydata, self.problemdata.bdrycond, self.diffcell)
                elif type == types[3]:
                    data['scalar'][name] = self.fem.computePointValues(u, colors)
                elif type == types[4]:
                    data['scalar'][name] = self.fem.computeMeanValues(u, colors)
                elif type == types[5]:
                    data['scalar'][name] = self.fem.computeLineValues(u, colors)
                else:
                    raise ValueError(f"unknown postprocess type '{type}' for key '{name}'\nknown types={types=}")
        return data

    def computeEstimator(self, u):
        if "rhs" in self.problemdata.params.fct_glob:
            xc, yc, zc = self.cell_coordinates_xyz()
            rhs_cell = self.problemdata.params.fct_glob["rhs"](xc, yc, zc)
        else:
            rhs_cell = np.zeros(self.mesh.ncells)

        eta, eta2 = self.fem.computeEstimator(
            u,
            rhs_cell=rhs_cell,
            diffcell=self.diffcell,
        )

        return SimpleNamespace(
            eta=eta,
            eta_cell=np.sqrt(eta2),
        )
