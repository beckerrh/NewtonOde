import numpy as np
from types import SimpleNamespace


# ======================================================================
class ArmijoGlobalization:
    def __init__(self, **kwargs):
        self.maxiter = kwargs.pop("maxiter", 20)
        self.omega = kwargs.pop("omega", 0.75)
        self.c = kwargs.pop("c", 1e-4)
        self.verbose = kwargs.pop("verbose", 1)
        self.relative_decrease = kwargs.pop("relative_decrease", True)
        self.log_types = {"ntrial": "i", "alpha": "e"}
        self.debug = kwargs.pop("debug", False)

    def accept(self, state, step, solver, info):
        x = step.x

        state_x = solver.nd.evaluate(x)

        print(
            "\nARMIJO BASE CHECK",
            "passed =", state.meritvalue,
            "eval(step.x) =", state_x.meritvalue,
            "ratio =", state_x.meritvalue / state.meritvalue,
        )

        if abs(state_x.meritvalue - state.meritvalue) > 1e-10 * max(1.0, abs(state.meritvalue)):
            raise RuntimeError(
                "Armijo inconsistency: "
                f"state.meritvalue={state.meritvalue}, "
                f"evaluate(step.x).meritvalue={state_x.meritvalue}"
            )

        alpha = 1.0

        phi0 = state.meritvalue

        for bt in range(self.maxiter):
            xtrial = solver.nd.add_update(x, alpha, step.dx)
            trial = solver.nd.evaluate(xtrial)

            if self.relative_decrease:
                meritvalue_aimed = (1.0 - self.c * alpha) * phi0
                accept = trial.meritvalue <= meritvalue_aimed
            else:
                meritvalue_aimed = phi0 + self.c * alpha * step.meritgrad
                accept = trial.meritvalue <= meritvalue_aimed

            if self.debug:
                print(
                    "alpha", alpha,
                    "phi0", state.meritvalue,
                    "trial", trial.meritvalue,
                    "trial2", trial.meritvalue ** 2,
                    "aimed", meritvalue_aimed,
                    "dx", step.dx_norm,
                    "xshape", x.shape,
                    "dxshape", step.dx.shape,
                )

            # print(
            #     "ARMIJO",
            #     "phi_ref", state.meritvalue,
            #     "phi_base_final", solver.nd.evaluate(x).meritvalue,
            #     "phi_trial", trial.meritvalue,
            #     "aimed", meritvalue_aimed,
            #     "alpha", alpha,
            # )
            #
            # print("dx_norm in armijo",
            #       step.dx.norm() if hasattr(step.dx, "norm") else np.linalg.norm(step.dx.flatten()))

            xtrial = solver.nd.add_update(x, 1.0, step.dx)
            # print("move_norm", np.linalg.norm(xtrial.flatten() - x.flatten()))

            if accept:
                return SimpleNamespace(
                    success=True,
                    x=xtrial,
                    state=trial,
                    alpha=alpha,
                    ntrial=bt,
                    aimed=meritvalue_aimed,
                    failure=None,
                )

            alpha *= self.omega

        return SimpleNamespace(
            success=False,
            x=x,
            state=state,
            alpha=alpha,
            ntrial=self.maxiter,
            aimed=meritvalue_aimed,
            failure="armijo backtracking failed",
        )
# class ArmijoGlobalization:
#     def __init__(self, **kwargs):
#         self.maxiter = kwargs.pop("maxiter", 20)
#         self.omega = kwargs.pop("omega", 0.75)
#         self.c = kwargs.pop("c", 1e-4)
#         self.verbose = kwargs.pop("verbose", 1)
#         self.relative_decrease = kwargs.pop("relative_decrease", False)
#         self.log_types = {"ntrial": "i", "alpha": "e"}
#     def accept(self, state, step, solver, info):
#         x = step.x
#         alpha = 1.0
#
#         for bt in range(self.maxiter):
#             xtrial = solver.nd.add_update(x, alpha, step.dx)
#             trial = solver.nd.evaluate(xtrial)
#
#             if self.relative_decrease:
#                 meritvalue_aimed = (1-self.c * alpha )*state.meritvalue
#             else:
#                 meritvalue_aimed = state.meritvalue + self.c * alpha * step.meritgrad
#             # print(f"{meritvalue_aimed=} {trial.meritvalue=}")
#             if trial.meritvalue <= meritvalue_aimed:
#                 return SimpleNamespace(
#                     success=True,
#                     x=xtrial,
#                     state=trial,
#                     alpha=alpha,
#                     ntrial=bt,
#                     aimed=meritvalue_aimed,
#                     failure=None,
#                 )
#
#             alpha *= self.omega
#
#         return SimpleNamespace(
#             success=False,
#             x=x,
#             state=state,
#             alpha=alpha,
#             ntrial=self.maxiter,
#             failure="armijo backtracking failed",
#         )

# ======================================================================
class WolfeGlobalization:
    def __init__(self, maxiter=20, omega=0.75, c1=1e-4, c2=0.9):
        self.maxiter = maxiter
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.log_types = {"ntrial": "i", "alpha": "e"}
    def accept(self, state, step, solver, info):
        x = step.x
        nd = solver.nd
        alpha = 1.0
        phi0 = state.meritvalue
        dphi0 = step.meritgrad

        for bt in range(self.maxiter):
            xtrial = nd.add_update(x, alpha, step.dx)
            trial = nd.evaluate(xtrial)

            armijo_ok = trial.meritvalue <= phi0 + self.c1 * alpha * dphi0

            dphia = nd.merit_directional_derivative(
                xtrial, trial, step.dx
            )

            curvature_ok = dphia >= self.c2 * dphi0
            # print(
            #     f"{bt=} {alpha=:.3e} "
            #     f"phi={trial.meritvalue:.3e} "
            #     f"aimed={phi0 + self.c1 * alpha * dphi0:.3e} "
            #     f"dphia={dphia:.3e} dphi0={dphi0:.3e} "
            #     f"armijo={armijo_ok} curvature={curvature_ok}"
            # )
            if armijo_ok and curvature_ok:
                return SimpleNamespace(
                    success=True,
                    x=xtrial,
                    state=trial,
                    alpha=alpha,
                    ntrial=bt,
                    aimed=phi0 + self.c1 * alpha * dphi0,
                    failure=None,
                )

            alpha *= self.omega

        return SimpleNamespace(
            success=False,
            x=x,
            state=state,
            alpha=alpha,
            ntrial=self.maxiter,
            aimed=phi0 + self.c1 * alpha * dphi0,
            failure="wolfe failed",
        )