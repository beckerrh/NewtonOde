import numpy as np
from scipy.sparse import block_diag
from types import SimpleNamespace

from .discretization_base import DiscretizationBase
from ..fems import cr1, p1
from ..linalg.fem_vector import FemVector

from .operators.convection_operator import ConvectionOperator
from .operators.diffusion_operator import DiffusionOperator
from .operators.reaction_operator import ReactionOperator

# ================================================================= #
class EllipticDiscretization(DiscretizationBase):

    def __repr__(self):
        lines = [
            f"{type(self).__name__}(",
            f"  fem={type(self.fem).__name__}",
            f"  mesh={self.mesh}",
            f"  ncomp={self.ncomps[0]}",
        ]

        for op in self.operators:
            for line in repr(op).splitlines():
                lines.append(f"  {line}")

        lines.append(f"  dirichlet={self.dirichletmethod}")

        if self.dirichletmethod == "nitsche":
            lines.append(f"  nitscheparam={self.nitscheparam}")
            lines.append(f"  nitsche_lumped={self.nitsche_lumped}")

        lines.append(")")
        return "\n".join(lines)

    def __init__(self, mesh, application, fem_name, disc_params, problemdata, **kwargs):

        self.rhs_fct = None

        if fem_name == "p1":
            fem = p1.P1()
        elif fem_name == "cr1":
            fem = cr1.CR1()
        else:
            raise NotImplementedError(f"{fem_name=}")

        fem.setMesh(mesh)

        verbose = kwargs.pop("verbose", False)
        timer = kwargs.pop("timer", None)

        super().__init__(
            mesh,
            application,
            fems=[fem],
            part_names=["U"],
            verbose=verbose,
            timer=timer,
        )

        self.problemdata = problemdata

        self.reaction_lumped = disc_params.pop("reaction_lumped", True)

        self.dirichletmethod = disc_params.pop("dirichletmethod", "nitsche")
        if self.dirichletmethod == "nitsche":
            self.nitscheparam = disc_params.pop("nitscheparam", 10)
            self.nitsche_lumped = disc_params.pop("nitsche_lumped", False)

        self.convectionmethod = disc_params.pop("convmethod", "lps")
        self.lpsparam = disc_params.pop("lpsparam", 0.2)

        self._checkProblemData()

        problemdata.params.begin_usage_tracking()
        self.prepareoperators(problemdata)
        # print("operators =", self.operators)


        msg = problemdata.params.unused_message()
        if msg:
            raise ValueError(f"*** unused problemdata params: {msg}")

        colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsflux = self.problemdata.postproc.colorsOfType("bdry_nflux")

        if self.dirichletmethod != "nitsche":
            self.bdrydata = self.fem.prepareBoundary(colorsdirichlet, colorsflux)

        assert len(self.application.ncomps) == 1

        if application.exactsolution is not None:
            self.generatePoblemDataForAnalyticalSolution()

        if kwargs:
            raise ValueError(f"*** unused arguments {kwargs=}")

        if disc_params:
            raise ValueError(f"*** unused arguments {disc_params=}")

    @property
    def diffusion(self):
        return self.get_operator(DiffusionOperator)

    @property
    def convection(self):
        return self.get_operator(ConvectionOperator)

    @property
    def reaction(self):
        return self.get_operator(ReactionOperator)
    # ================================================================= #
    def prepareoperators(self, problemdata, debug=False):
        diffusion = DiffusionOperator.from_problemdata(
                problemdata, self.mesh, self.fem, self.ncomps[0]
            )

        self.add_operator("U", "U", diffusion)

        convection = ConvectionOperator.from_problemdata(
            problemdata=problemdata,
            mesh=self.mesh,
            fem=self.fem,
            ncomp=self.ncomps[0],
            method=self.convectionmethod,
            lpsparam=self.lpsparam,
        )
        if convection is not None:
            self.add_operator("U", "U", convection)

        reaction = ReactionOperator.from_problemdata(
            problemdata=problemdata,
            mesh=self.mesh,
            fem=self.fem,
            ncomp=self.ncomps[0],
            lumped=self.reaction_lumped,
        )
        if reaction is not None:
            self.add_operator("U", "U", reaction)

        if debug:
            print("scal_glob =", problemdata.params.scal_glob)
            print("fct_glob =", problemdata.params.fct_glob)
            print("lookup reaction =", problemdata.params.lookup(
                "reaction",
                ("fct_glob", "scal_glob", "scal_cells", "scal_celllabels", "data"),
            ))

    def boundary_dofs_global(self, bdrydata=None, ncomp=None):
        if bdrydata is None:
            bdrydata = self.bdrydata
        if ncomp is None:
            ncomp = self.ncomps[0]

        bd = np.asarray(self.fem.dirichlet_dofs(bdrydata), dtype=int)
        ndof = self.fem.nunknowns()

        if ncomp == 1:
            return bd

        return np.concatenate([
            c * ndof + bd
            for c in range(ncomp)
        ])

    def add_boundary_form(self, du, u):
        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")

        diffusion = self.get_operator(DiffusionOperator)
        if diffusion is None:
            raise RuntimeError("DiffusionOperator is missing")

        U = u.part("U")
        DU = du.part("U")
        ncomp = U.shape[0]

        for icomp in range(ncomp):
            ui = U[icomp]
            dui = DU[icomp]

            self.fem.massDotBoundary(
                dui,
                ui,
                colorsrobin,
                bdrycond.param,
                lumped=True,
            )

            if self.dirichletmethod != "nitsche":
                bd = self.fem.dirichlet_dofs(self.bdrydata)
                dui[bd] = 0.0
            else:
                diff_i = diffusion.component_coeff(icomp)

                self.fem.computeFormNitscheDiffusion(
                    self.nitscheparam,
                    dui,
                    ui,
                    diff_i,
                    colorsdir,
                    lumped=self.nitsche_lumped,
                )
    def _repeat_scalar_matrix(self, A0):
        ncomp = self.ncomps[0]
        if ncomp == 1:
            return A0
        return block_diag([A0] * ncomp, format="csr")

    def add_boundary_matrix(self, A, u=None):
        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")

        ncomp = self.ncomps[0]

        # 1. Robin contribution
        Rb = self.fem.computeBdryMassMatrix(
            colorsrobin,
            bdrycond.param,
            lumped=True,
        )
        if Rb is not None:
            A += self._repeat_scalar_matrix(Rb)

        # 2. Nitsche contribution, only for nitsche
        if self.dirichletmethod == "nitsche":
            diffusion = self.get_operator(DiffusionOperator)
            if diffusion is None:
                raise RuntimeError("DiffusionOperator is missing")

            from scipy.sparse import block_diag

            blocks = []
            for icomp in range(ncomp):
                diff_i = diffusion.component_coeff(icomp)

                blocks.append(
                    self.fem.computeMatrixNitscheDiffusion(
                        self.nitscheparam,
                        diffcoff=diff_i,
                        colors=colorsdir,
                        lumped=self.nitsche_lumped,
                    )
                )

            A += block_diag(blocks, format="csr")

        # 3. Strong Dirichlet elimination LAST
        elif self.dirichletmethod == "strong":
            from ..linalg.constraints import eliminate_matrix_symmetric

            bd = self.boundary_dofs_global()
            A = eliminate_matrix_symmetric(A, bd)

        else:
            raise ValueError(f"unknown {self.dirichletmethod=}")

        return A
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
        ncomp = self.ncomps[0]
        dim = self.mesh.dimension

        if len(solexact) != ncomp:
            raise ValueError(f"{len(solexact)=} != {ncomp=}")

        diffusion = self.get_operator(DiffusionOperator)
        convection = self.find_operator(ConvectionOperator)
        reaction = self.find_operator(ReactionOperator)

        def rhs(x, y, z):
            z = np.zeros_like(x) if z is None else z
            out = np.zeros((ncomp, x.size))

            # diffusion: -div(k_i grad u_i)
            for i, ui in enumerate(solexact):
                ki = diffusion.coeff_pointwise(i)

                for d in range(dim):
                    out[i] -= ki * ui.dd(d, d, x, y, z)

            # convection: beta_i · grad u_i
            if convection is not None:
                convection.add_to_manufactured_rhs(out, solexact, x, y, z)

            # reaction: sum_j c_ij u_j
            if reaction is not None:
                reaction.add_to_manufactured_rhs(out, solexact, x, y, z)

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

        # self.problemdata.params.fct_glob["rhs"] = \
        #     self.defineRhsAnalyticalSolution(solexact)
        self.rhs_fct = self.defineRhsAnalyticalSolution(solexact)

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

    def apply_strong_dirichlet_to_solution(self, u):
        bdrycond = self.problemdata.bdrycond
        colorsdir = bdrycond.colorsOfType("Dirichlet")
        U = u.part("U")
        ncomp = U.shape[0]

        for icomp in range(ncomp):
            bdrycondfct_i = {
                color: self._component_boundary_function(
                    bdrycond.fct[color],
                    icomp,
                )
                for color in colorsdir
            }

            self.fem.vectorBoundaryStrong(
                U[icomp],
                SimpleNamespace(fct=bdrycondfct_i),
                self.bdrydata,
            )

        return u

    def initial_guess(self):
        if getattr(self, "u0", None) is not None:
            u0 = self.u0
            self.u0 = None
            return u0.copy()

        u0 = self.newVector()

        if self.dirichletmethod == "strong":
            self.apply_strong_dirichlet_to_solution(u0)

        return u0

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

        B = b.part(0)

        if B.shape != (ncomp, ndof):
            raise ValueError(f"{B.shape=} expected {(ncomp, ndof)}")

        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdir = bdrycond.colorsOfType("Dirichlet")
        colorsneu = bdrycond.colorsOfType("Neumann")
        if self.rhs_fct is not None:
            fp1 = np.asarray(self.fem.interpolate(self.rhs_fct))

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
            diffusion = self.get_operator(DiffusionOperator)
            if diffusion is None:
                raise RuntimeError("DiffusionOperator is missing")
            for icomp in range(ncomp):
                bdrycondfct_i = {
                    color: self._component_boundary_function(bdrycond.fct[color], icomp)
                    for color in colorsdir
                }

                diff_i = diffusion.component_coeff(icomp)

                self.fem.computeRhsNitscheDiffusion(
                    self.nitscheparam,
                    B[icomp],
                    diff_i,
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
        convection = self.find_operator(ConvectionOperator)

        if convection is not None:
            bdrycondfct_by_comp = []

            for icomp in range(ncomp):
                bdrycondfct_i = {
                    color: self._component_boundary_function(
                        bdrycond.fct[color],
                        icomp,
                    )
                    for color in colorsdir
                }
                bdrycondfct_by_comp.append(bdrycondfct_i)

            convection.add_dirichlet_inflow_rhs(
                self,
                B,
                colorsdir,
                bdrycondfct_by_comp,
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
        data = {
            "scalar": {},
            "cell": {},
        }
        U = u.part("U")
        ncomp = U.shape[0]

        if self.application.exactsolution:
            scal, cell = self.compute_errors_exact(u)
            data["scalar"].update(scal)
            data["cell"] = cell

        if self.problemdata.postproc:

            diffusion = self.get_operator(DiffusionOperator)
            if diffusion is None:
                raise RuntimeError("DiffusionOperator is missing")

            types = [
                "bdry_mean",
                "bdry_fct",
                "bdry_nflux",
                "pointvalues",
                "meanvalues",
                "linemeans",
            ]

            for icomp in range(ncomp):

                ui = U[icomp]
                diff_i = diffusion.component_coeff(icomp)

                for name, type in self.problemdata.postproc.type.items():

                    key = name if ncomp == 1 else f"U{icomp}_{name}"

                    colors = self.problemdata.postproc.colors(name)

                    if type == types[0]:
                        data["scalar"][key] = self.fem.computeBdryMean(
                            ui,
                            colors,
                        )

                    elif type == types[1]:
                        data["scalar"][key] = self.fem.computeBdryFct(
                            ui,
                            colors,
                        )

                    elif type == types[2]:

                        if self.dirichletmethod == "nitsche":

                            bdrycondfct_i = {
                                color: self._component_boundary_function(
                                    self.problemdata.bdrycond.fct[color],
                                    icomp,
                                )
                                for color in colors
                            }

                            udir = self.fem.interpolateBoundary(
                                colors,
                                bdrycondfct_i,
                            )

                            data["scalar"][key] = (
                                self.fem.computeBdryNormalFluxNitsche(
                                    self.nitscheparam,
                                    ui,
                                    colors,
                                    udir,
                                    diff_i,
                                )
                            )

                        else:
                            data["scalar"][key] = (
                                self.fem.computeBdryNormalFlux(
                                    ui,
                                    colors,
                                    self.bdrydata,
                                    self.problemdata.bdrycond,
                                    diff_i,
                                )
                            )

                    elif type == types[3]:
                        data["scalar"][key] = self.fem.computePointValues(
                            ui,
                            colors,
                        )

                    elif type == types[4]:
                        data["scalar"][key] = self.fem.computeMeanValues(
                            ui,
                            colors,
                        )

                    elif type == types[5]:
                        data["scalar"][key] = self.fem.computeLineValues(
                            ui,
                            colors,
                        )

                    else:
                        raise ValueError(
                            f"unknown postprocess type '{type}' for key '{name}'\n"
                            f"known types={types=}"
                        )
        return data

    def computeEstimator(self, u, du=None):
        import numpy as np
        from types import SimpleNamespace

        U = u.part("U")
        ncomp = U.shape[0]

        DU = None
        if du is not None:
            DU = du.part("U")
            if DU.shape != U.shape:
                raise ValueError(f"{DU.shape=} != {U.shape=}")

        diffusion = self.get_operator(DiffusionOperator)
        if diffusion is None:
            raise RuntimeError("DiffusionOperator is missing")

        if diffusion.kind == "coupled":
            raise NotImplementedError("Estimator for coupled diffusion")

        xc, yc, zc = self.cell_coordinates_xyz()

        if self.rhs_fct is not None:
            rhs_base = np.asarray(self.rhs_fct(xc, yc, zc))
            if rhs_base.ndim == 1:
                rhs_base = rhs_base.reshape(1, -1)
        else:
            rhs_base = np.zeros((ncomp, self.mesh.ncells))

        if rhs_base.shape != (ncomp, self.mesh.ncells):
            raise ValueError(
                f"{rhs_base.shape=} expected {(ncomp, self.mesh.ncells)}"
            )

        convection = self.find_operator(ConvectionOperator)
        reaction = self.find_operator(ReactionOperator)

        skip_faces = None
        if self.dirichletmethod == "strong":
            skip_faces = self.fem.boundary_faces_from_bdrydata(self.bdrydata)

        def estimate2_cell(V, rhs_cell):
            eta2_total = np.zeros(self.mesh.ncells)

            for i in range(ncomp):
                diff_i = diffusion.estimator_coeff(i)

                _, eta2_i = self.fem.computeEstimator(
                    V[i],
                    rhs_cell=rhs_cell[i],
                    diffcell=diff_i,
                    skip_faces=skip_faces,
                )

                eta2_total += eta2_i

            return eta2_total

        # eta(u): estimator of the current nonlinear/discrete residual
        rhs_eta = rhs_base.copy()

        if convection is not None:
            rhs_eta = convection.subtract_from_cell_rhs(self, rhs_eta, U)

        if reaction is not None:
            rhs_eta = reaction.subtract_from_cell_rhs(self, rhs_eta, U)

        eta2_cell = estimate2_cell(U, rhs_eta)
        eta = float(np.sqrt(np.sum(eta2_cell)))

        if DU is None:
            return SimpleNamespace(
                eta=eta,
                eta2_cell=eta2_cell,
            )

        # zeta(u, du): estimator of F(u) + F'(u)du
        rhs_zeta = rhs_eta.copy()

        if convection is not None:
            rhs_zeta = convection.subtract_from_cell_rhs(self, rhs_zeta, U+DU)

        if reaction is not None:
            rhs_zeta = reaction.subtract_linearized_from_cell_rhs(
                self, rhs_zeta, U, DU
            )

        zeta2_cell = estimate2_cell(U+DU, rhs_zeta)
        zeta = float(np.sqrt(np.sum(zeta2_cell)))

        return SimpleNamespace(
            eta=eta,
            eta2_cell=eta2_cell,
            zeta=zeta,
            zeta2_cell=zeta2_cell,
        )

    def plot_data(self, u, eta=None):
        U = u.part("U")
        ui = U[0]

        point_u = self.fem.to_p1(ui)

        cell = {}

        diffusion = self.get_operator(DiffusionOperator)
        if diffusion is not None:
            cell.update(diffusion.plot_cell_data())

        if eta is not None:
            cell["eta"] = eta

        return {
            "point": {"u": point_u},
            "cell": cell,
            "global": {},
        }