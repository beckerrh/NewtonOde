import numpy as np
from scipy.sparse import block_diag, bmat

from types import SimpleNamespace

from Utility.analyticalfunction import AnalyticalFunction
from .discretization_base import DiscretizationBase
from ..fems import cr1, p1, rt0, mesh_transfer
from ..fems import data as femdata
from ..fems.diffusion import normalize_diffusion
from ..fems.reaction import normalize_reaction
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

        if "reaction" in self.problemdata.params.scal_glob:
            reaction = self.problemdata.params.scal_glob["reaction"]
        elif "reaction" in self.problemdata.params.scal_cells:
            reaction = self.compute_cell_vector_from_params("reaction", self.problemdata.params)
        else:
            reaction = None
        self.reaction = reaction
        if reaction is None:
            self.reactionkind = None
            self.reactioncell = None
        else:
            self.reactionkind, self.reactioncell = normalize_reaction(
                reaction,
                self.mesh.ncells,
                ncomp=self.ncomps[0],
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

        if application.exactsolution is not None:
            self.generatePoblemDataForAnalyticalSolution()

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
    # def defineRhsAnalyticalSolution(self, solexact_list):
    #     solexact = solexact_list[0]
    #     ncomp = self.ncomps[0]
    #     dim = self.mesh.dimension
    #     kheat = self.problemdata.params.scal_glob["kheat"]
    #
    #     def rhs(x, y, z):
    #         z = np.zeros_like(x) if z is None else z
    #         out = np.zeros((ncomp, x.size))
    #
    #         for icomp, ui in enumerate(solexact):
    #             # diffusion: -k Δu_i
    #             for d in range(dim):
    #                 out[icomp] -= kheat * ui.dd(d, d, x, y, z)
    #
    #             # convection: beta · grad u_i
    #             if self.hasconvection:
    #                 for d in range(dim):
    #                     out[icomp] += (
    #                             self.convection_fct[d](x, y, z)
    #                             * ui.d(d, x, y, z)
    #                     )
    #
    #         return out
    #
    #     return rhs
    def defineRhsAnalyticalSolution(self, solexact_list):
        solexact = solexact_list[0]  # tuple of component exact solutions
        ncomp = self.ncomps[0]
        dim = self.mesh.dimension

        if len(solexact) != ncomp:
            raise ValueError(f"{len(solexact)=} != {ncomp=}")

        if self.diffkind != "scalar":
            raise NotImplementedError(
                "Manufactured RHS currently supports only scalar constant diffusion"
            )

        kheat = self.problemdata.params.scal_glob["kheat"]

        def rhs(x, y, z):
            z = np.zeros_like(x) if z is None else z
            out = np.zeros((ncomp, x.size))

            for i, ui in enumerate(solexact):
                # diffusion: -k Δu_i
                for d in range(dim):
                    out[i] -= kheat * ui.dd(d, d, x, y, z)

                # convection: beta · grad u_i
                if self.hasconvection:
                    for d in range(dim):
                        out[i] += (
                                self.convection_fct[d](x, y, z)
                                * ui.d(d, x, y, z)
                        )

                # reaction: sum_j C_ij u_j
                if getattr(self, "reactioncell", None) is not None:
                    for j, uj in enumerate(solexact):
                        cij = self.reaction_coeff_pointwise(i, j)
                        # cij = self.reaction_coeff(i, j)

                        if np.isscalar(cij):
                            if cij == 0:
                                continue
                            out[i] += cij * uj(x, y, z)
                        else:
                            raise NotImplementedError(
                                "Manufactured RHS with cellwise reaction is not pointwise-defined"
                            )

            return out

        return rhs
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
        U = u.block("U")
        ncomp = U.shape[0]

        du = u.zeros_like()
        DU = du.block("U")

        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")

        for icomp in range(ncomp):
            ui = U[icomp]
            dui = DU[icomp]
            diff_i = self._scalar_component_diffusion(icomp)

            self.fem.computeFormDiffusion(dui, ui, diff_i)

            if self.hasconvection:
                self.fem.computeFormTransportCellWise(
                    dui,
                    ui,
                    self.convdata,
                    type="centered",
                )
                if hasattr(self.fem, "computeFormJump"):
                    self.fem.computeFormJump(dui, ui, self.convdata.betart)
                if self.convectionmethod == "lps":
                    self.fem.computeFormLps(
                        dui,
                        ui,
                        self.convdata.betart,
                        lpsparam=self.lpsparam,
                    )

            if coeffmass is not None:
                self.fem.massDot(dui, ui, coeff=coeffmass)

            self.fem.massDotBoundary(
                dui,
                ui,
                colorsrobin,
                bdrycond.param,
                lumped=True,
            )

            if self.dirichletmethod != "nitsche":
                self.fem.vectorBoundaryStrongEqual(dui, ui, self.bdrydata)
            else:
                self.fem.computeFormNitscheDiffusion(
                    self.nitscheparam,
                    dui,
                    ui,
                    diff_i,
                    colorsdir,
                    lumped=self.nitsche_lumped,
                )

        return du
    def _repeat_scalar_matrix(self, A0):
        ncomp = self.ncomps[0]
        if ncomp == 1:
            return A0
        return block_diag([A0] * ncomp, format="csr")

    def reaction_coeff_pointwise(self, i, j):
        if getattr(self, "reaction", None) is None:
            return 0.0

        r = np.asarray(self.reaction, dtype=float)

        if r.ndim == 0:
            return float(r) if i == j else 0.0

        if r.shape == (self.ncomps[0],):
            return float(r[i]) if i == j else 0.0

        if r.shape == (self.ncomps[0], self.ncomps[0]):
            return float(r[i, j])

        raise NotImplementedError(
            "Manufactured RHS only supports constant scalar/diagonal/matrix reaction"
        )
    def reaction_coeff(self, i, j):
        if getattr(self, "reactioncell", None) is None:
            return 0.0

        if self.reactionkind == "scalar":
            return self.reactioncell if i == j else 0.0

        if self.reactionkind == "diagonal":
            return self.reactioncell[i] if i == j else 0.0

        if self.reactionkind == "coupled":
            return self.reactioncell[i, j]

        raise ValueError(f"unknown {self.reactionkind=}")
    def _scalar_component_diffusion(self, icomp):
        if self.diffkind == "scalar":
            return self.diffcell

        if self.diffkind == "diagonal":
            return self.diffcell[icomp]

        raise ValueError(
            "Scalar component diffusion requested for coupled diffusion"
        )

    def _add_reaction_scalar_diagonal(self, Ai, icomp):
        if self.reactionkind is None:
            return Ai

        if self.reactionkind == "scalar":
            return Ai + self.fem.computeMassMatrix(coeff=self.reactioncell)

        if self.reactionkind == "diagonal":
            return Ai + self.fem.computeMassMatrix(coeff=self.reactioncell[icomp])

        return Ai

    def _reaction_matrix(self):
        from scipy.sparse import block_diag, bmat

        if self.reactioncell is None:
            return None

        ncomp = self.ncomps[0]

        if self.reactionkind == "scalar":
            R0 = self.fem.computeMassMatrix(coeff=self.reactioncell)
            return block_diag([R0] * ncomp, format="csr")

        if self.reactionkind == "diagonal":
            return block_diag(
                [
                    self.fem.computeMassMatrix(coeff=self.reactioncell[i])
                    for i in range(ncomp)
                ],
                format="csr",
            )

        if self.reactionkind == "coupled":
            return bmat(
                [
                    [
                        self.fem.computeMassMatrix(coeff=self.reactioncell[i, j])
                        for j in range(ncomp)
                    ]
                    for i in range(ncomp)
                ],
                format="csr",
            )

        raise ValueError(f"unknown {self.reactionkind=}")

    def computeMatrix(self, u=None, coeffmass=None):
        from scipy.sparse import block_diag, bmat

        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")

        ncomp = self.ncomps[0]

        def add_scalar_lower_order(Ai):
            Ai += self.fem.computeBdryMassMatrix(
                colorsrobin,
                bdrycond.param,
                lumped=True,
            )

            if self.hasconvection:
                Ai += self.fem.computeMatrixTransportCellWise(
                    self.convdata,
                    type="centered",
                )

                if hasattr(self.fem, "computeMatrixJump"):
                    Ai += self.fem.computeMatrixJump(self.convdata.betart)

                if self.convectionmethod == "lps":
                    Ai += self.fem.computeMatrixLps(
                        self.convdata.betart,
                        lpsparam=self.lpsparam,
                    )

            if coeffmass is not None:
                Ai += self.fem.computeMassMatrix(coeff=coeffmass)

            return Ai

        def reaction_matrix():
            if getattr(self, "reactioncell", None) is None:
                return None

            if self.reactionkind == "scalar":
                R0 = self.fem.computeMassMatrix(coeff=self.reactioncell)
                return block_diag([R0] * ncomp, format="csr")

            if self.reactionkind == "diagonal":
                return block_diag(
                    [
                        self.fem.computeMassMatrix(coeff=self.reactioncell[i])
                        for i in range(ncomp)
                    ],
                    format="csr",
                )

            if self.reactionkind == "coupled":
                return bmat(
                    [
                        [
                            self.fem.computeMassMatrix(
                                coeff=self.reactioncell[i, j]
                            )
                            for j in range(ncomp)
                        ]
                        for i in range(ncomp)
                    ],
                    format="csr",
                )

            raise ValueError(f"unknown {self.reactionkind=}")

        # ------------------------------------------------------------
        # scalar/diagonal diffusion
        # ------------------------------------------------------------
        if self.diffkind in ("scalar", "diagonal"):
            blocks = []

            for icomp in range(ncomp):
                diff_i = self._scalar_component_diffusion(icomp)

                Ai = self.fem.computeMatrixDiffusion(diff_i)
                Ai = add_scalar_lower_order(Ai)

                if self.dirichletmethod != "nitsche":
                    Ai = self.fem.matrixBoundaryStrong(Ai, self.bdrydata)
                else:
                    Ai += self.fem.computeMatrixNitscheDiffusion(
                        self.nitscheparam,
                        diffcoff=diff_i,
                        colors=colorsdir,
                        lumped=self.nitsche_lumped,
                    )

                blocks.append(Ai)

            A = block_diag(blocks, format="csr")

            R = reaction_matrix()
            if R is not None:
                if (
                        self.dirichletmethod != "nitsche"
                        and self.reactionkind == "coupled"
                ):
                    raise NotImplementedError(
                        "strong Dirichlet with coupled reaction needs "
                        "global block boundary treatment"
                    )
                A += R

            return A

        # ------------------------------------------------------------
        # coupled diffusion
        # ------------------------------------------------------------
        if self.diffkind == "coupled":
            matrix_blocks = []

            for i in range(ncomp):
                row = []
                for j in range(ncomp):
                    kij = self.diffcell[i, j]

                    if np.all(kij == 0):
                        row.append(None)
                        continue

                    Aij = self.fem.computeMatrixDiffusion(kij)

                    if i == j:
                        Aij = add_scalar_lower_order(Aij)

                        if self.dirichletmethod != "nitsche":
                            Aij = self.fem.matrixBoundaryStrong(
                                Aij,
                                self.bdrydata,
                            )
                        else:
                            raise NotImplementedError(
                                "Nitsche matrix for coupled diffusion is not implemented yet"
                            )

                    row.append(Aij)

                matrix_blocks.append(row)

            A = bmat(matrix_blocks, format="csr")

            R = reaction_matrix()
            if R is not None:
                if self.dirichletmethod != "nitsche":
                    raise NotImplementedError(
                        "strong Dirichlet with coupled diffusion/reaction needs "
                        "global block boundary treatment"
                    )
                A += R

            return A

        raise ValueError(f"unknown {self.diffkind=}")
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
            for icomp in range(ncomp):
                bdrycondfct_i = {
                    color: self._component_boundary_function(
                        bdrycond.fct[color],
                        icomp,
                    )
                    for color in colorsdir
                }

                self.fem.vectorBoundaryStrong(
                    B[icomp],
                    SimpleNamespace(fct=bdrycondfct_i),
                    self.bdrydata,
                )
        if self.hasconvection:
            for icomp in range(ncomp):
                bdrycondfct_i = {
                    color: self._component_boundary_function(
                        bdrycond.fct[color],
                        icomp,
                    )
                    for color in colorsdir
                }

                fp1_i = self.fem.interpolateBoundary(colorsdir, bdrycondfct_i)

                self.fem.massDotBoundary(
                    B[icomp],
                    fp1_i,
                    colors=colorsdir,
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
        if hasattr(self, "bdrydata"):
            for icomp in range(ncomp):
                bdrycondfct_i = {
                    color: self._component_boundary_function(
                        bdrycond.fct[color],
                        icomp,
                    )
                    for color in colorsdir
                }

                self.fem.vectorBoundaryStrong(
                    B[icomp],
                    SimpleNamespace(fct=bdrycondfct_i),
                    self.bdrydata,
                )
        return b

    def postProcess(self, u):
        data = {"scalar": {}}

        U = u.block("U")
        ncomp = U.shape[0]

        if self.application.exactsolution:
            scal, cell = self.compute_errors_exact(u)
            data["scalar"].update(scal)
            data["cell"] = cell

        if self.problemdata.postproc:
            if ncomp != 1:
                raise NotImplementedError(
                    "postproc boundary/point quantities for vector-valued U "
                    "need componentwise naming"
                )

            ui = U[0]

            types = [
                "bdry_mean",
                "bdry_fct",
                "bdry_nflux",
                "pointvalues",
                "meanvalues",
                "linemeans",
            ]

            for name, type in self.problemdata.postproc.type.items():
                colors = self.problemdata.postproc.colors(name)

                if type == types[0]:
                    data["scalar"][name] = self.fem.computeBdryMean(ui, colors)

                elif type == types[1]:
                    data["scalar"][name] = self.fem.computeBdryFct(ui, colors)

                elif type == types[2]:
                    if self.dirichletmethod == "nitsche":
                        udir = self.fem.interpolateBoundary(
                            colors,
                            self.problemdata.bdrycond.fct,
                        )
                        data["scalar"][name] = self.fem.computeBdryNormalFluxNitsche(
                            self.nitscheparam,
                            ui,
                            colors,
                            udir,
                            self._scalar_component_diffusion(0),
                        )
                    else:
                        data["scalar"][name] = self.fem.computeBdryNormalFlux(
                            ui,
                            colors,
                            self.bdrydata,
                            self.problemdata.bdrycond,
                            self._scalar_component_diffusion(0),
                        )

                elif type == types[3]:
                    data["scalar"][name] = self.fem.computePointValues(ui, colors)

                elif type == types[4]:
                    data["scalar"][name] = self.fem.computeMeanValues(ui, colors)

                elif type == types[5]:
                    data["scalar"][name] = self.fem.computeLineValues(ui, colors)

                else:
                    raise ValueError(
                        f"unknown postprocess type '{type}' for key '{name}'\n"
                        f"known types={types=}"
                    )

        return data
    def computeEstimator(self, u):
        U = u.block("U")
        ncomp = U.shape[0]

        if self.diffkind == "coupled":
            raise NotImplementedError("Estimator for coupled diffusion")

        xc, yc, zc = self.cell_coordinates_xyz()

        if "rhs" in self.problemdata.params.fct_glob:
            rhs_cell = np.asarray(self.problemdata.params.fct_glob["rhs"](xc, yc, zc))
            if rhs_cell.ndim == 1:
                rhs_cell = rhs_cell.reshape(1, -1)
        else:
            rhs_cell = np.zeros((ncomp, self.mesh.ncells))

        # subtract convection: beta · grad u_i
        if self.hasconvection:
            beta = self.convdata.betacell[:, :self.mesh.dimension]
            for i in range(ncomp):
                grad_i = self.fem.cell_grad(U[i])
                rhs_cell[i] -= np.einsum("nd,nd->n", beta, grad_i)

        # subtract reaction: sum_j c_ij u_j
        if getattr(self, "reactioncell", None) is not None:
            Ucell = np.vstack([self.fem.to_cell(U[j]) for j in range(ncomp)])

            for i in range(ncomp):
                for j in range(ncomp):
                    cij = self.reaction_coeff(i, j)
                    if np.isscalar(cij) and cij == 0:
                        continue
                    rhs_cell[i] -= cij * Ucell[j]

        eta2_total = np.zeros(self.mesh.ncells)

        for i in range(ncomp):
            diff_i = self._scalar_component_diffusion(i)

            _, eta2_i = self.fem.computeEstimator(
                U[i],
                rhs_cell=rhs_cell[i],
                diffcell=diff_i,
            )
            eta2_total += eta2_i

        return SimpleNamespace(
            eta=np.sqrt(np.sum(eta2_total)),
            eta_cell=np.sqrt(eta2_total),
        )

    # in EllipticDiscretization
    def plot_data(self, u, eta=None):
        U = u.block("U")
        ui = U[0]

        if hasattr(self.fem, "to_p1"):
            point_u = self.fem.to_p1(ui)
        else:
            point_u = ui

        cell = {"k": self.kheatcell}
        if eta is not None:
            cell["eta"] = eta

        return {
            "point": {"u": point_u},
            "cell": cell,
            "global": {},
        }