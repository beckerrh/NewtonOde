import numpy as np
from types import SimpleNamespace

from scipy.signal import kaiser_beta
from torch.backends.mkl import verbose

from .base_newton_driver import BaseNewtonDriver
from ..mesh import marking, mesh_hierarchy
import Utility



#=================================================================#
class NewtonDriverAFEM(BaseNewtonDriver):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.beta = kwargs.pop('beta', 0.5)
        self.theta = kwargs.pop('theta', 0.9)
        self.max_meshiter = kwargs.pop('max_meshiter', 20)
        self.max_cells = kwargs.pop('max_cells', 10e6)
        self.logger_afem = Utility.logger.Logger(
            verbose=self.verbose,
            types={
                "it": "d:3",
                "meshit": "d:3",
                "N": "d:7",
                "eta": "e",
                "zeta": "e",
                "aimed": "e",
                "ratio": "e",
                "|p|": "e",
                "|F_h|": "e",
            },
            name="afem"
        )

    def attach_logger(self, logger):
        self.logger_newton = logger
        self.logger_newton.add_types({
            "eta": "e",
            "|F_h|": "e",
            "N": "d:7",
            "meshiter": "d:3",
        })
        self.logger_newton.update(N=self.model.discs[-1].mesh.ncells)

    def evaluate(self, x):
        from types import SimpleNamespace
        import numpy as np

        disc = self.model.discs[-1]

        F = disc.computeForm(x)
        b = disc.computeRhs()
        r = F.flatten() - b.flatten()

        if disc.dirichletmethod == "strong":
            bd = disc.boundary_dofs_global()
            r[bd] = 0.0

        resn = np.linalg.norm(r)

        est = disc.computeEstimator(x)

        merit = np.sqrt(est.eta ** 2 + self.beta * resn ** 2)

        self.logger_newton.update(
            eta=est.eta,
            N=disc.mesh.ncells,
        )
        self.logger_newton.values["|F_h|"] = resn

        return SimpleNamespace(
            meritvalue=merit,
            residual=r,
            residual_norm=resn,
            eta=est.eta,
            norm_X=np.linalg.norm(x.flatten()),
        )
    def compute_newton_step(self, x, state, info, debug=False):
        from types import SimpleNamespace
        import numpy as np

        aimed = info.tol_linear_abs

        liniter_total = 0
        self.logger_afem.print_names()
        for meshiter in range(self.max_meshiter):
            disc = self.model.discs[-1]

            # residual must live on the current mesh
            r = state.residual

            A = disc.computeMatrix(x)
            if len(self.model.As) == len(self.model.discs):
                self.model.As[-1] = A
            elif len(self.model.As) == len(self.model.discs) - 1:
                self.model.As.append(A)
            else:
                raise RuntimeError(...)

            if self.debug:
                from scipy.sparse.linalg import eigsh

                Asym = A - A.T
                asym = np.linalg.norm(Asym.data) / max(1.0, np.linalg.norm(A.data))
                print("matrix asymmetry", asym)

                S = 0.5 * (A + A.T)

                for qid in range(5):
                    q = np.random.randn(S.shape[0])
                    Sq = S @ q
                    print("sym rayleigh", q @ Sq / (q @ q))

                if A.shape[0] < 2000:
                    S = 0.5 * (A + A.T)
                    lam_min = np.linalg.eigvalsh(S.toarray())[0]
                    print("lambda_min symmetric part", lam_min)

                self.check_jacobian_fd(x)

            self.model.B.update(A=A)

            p0 = getattr(disc, "p0", None)

            if p0 is None:
                x0_lin = np.zeros_like(r)
            else:
                x0_lin = p0.flatten()

            # print("solver type", type(self.model.B))
            # print("solver dict", getattr(self.model.B, "__dict__", {}))

            p_flat = self.model.B.solve(b=-r, x0=x0_lin)

            p = x.from_flat_like(p_flat)

            liniter = getattr(self.model.B, "niter", -1)
            liniter_total += liniter


            disc.p0 = p

            pnorm = np.linalg.norm(p_flat)

            # linearized/tangent estimator
            est = disc.computeEstimator(x, du=p)

            indicators = est.zeta2_cell

            ratio = est.zeta / max(aimed, 1e-300)

            self.logger_afem.update(
                it=info.it,
                meshit=meshiter,
                N=disc.mesh.ncells,
                eta=est.eta,
                zeta=est.zeta,
                aimed=aimed,
                ratio=ratio,
                **{"|p|": pnorm},
                **{"|F_h|": state.residual_norm},
            )

            self.logger_afem.print()


            if est.zeta <= aimed:
                self.logger_newton.update(
                    N=disc.mesh.ncells,
                    meshiter=meshiter,
                )

                disc.u0 = x

                # print("return pnorm", pnorm)
                # print("Ap+r", np.linalg.norm(A @ p.flatten() + r))

                if p.shape != x.shape:
                    raise ValueError(f"Newton step mesh mismatch: {x.shape=} {p.shape=}")
                return SimpleNamespace(
                    dx=p,
                    dx_norm=pnorm,
                    liniter=liniter_total,
                    x=x,  # base point on accepted mesh
                    success=True,
                )

            if disc.mesh.ncells > self.max_cells:
                return SimpleNamespace(
                    dx=x.zeros_like(),
                    dx_norm=0.0,
                    liniter=liniter_total,
                    x=x,
                    success=False,
                    failure="max_cells reached",
                )
            marked = marking.dorfler_marking(indicators, theta=self.theta)

            mesh2, refinfo = self.model.mesh_hierarchy.refine_nvb(marked)

            disc2 = self.model.discretize(mesh2)
            self.model.discs.append(disc2)

            transfer = disc.build_transfer_to_refined_mesh(
                refinfo,
                disc_fine=disc2,
            )
            self.model.transfers.append(transfer)

            assert len(self.model.As) == len(self.model.discs) - 1
            assert len(self.model.transfers) == len(self.model.discs) - 1

            x = transfer.interpolate(x)
            p0 = transfer.interpolate(p)

            disc2.u0 = x
            disc2.p0 = p0

            # residual/merit must be recomputed on refined mesh
            state = self.evaluate(x)
            continue

        # failed tangent AFEM solve; return last computed compatible step
        disc = self.model.discs[-1]
        disc.u0 = x

        return SimpleNamespace(
            dx=x.zeros_like(),
            dx_norm=0.0,
            liniter=liniter_total,
            x=x,
            success=False,
            failure="tangent AFEM did not produce compatible step",
        )