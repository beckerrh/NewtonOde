from types import SimpleNamespace


# ======================================================================
class ArmijoGlobalization:
    def __init__(self, **kwargs):
        self.maxiter = kwargs.pop("maxiter", 20)
        self.omega = kwargs.pop("omega", 0.75)
        self.c = kwargs.pop("c", 1e-4)
        self.verbose = kwargs.pop("verbose", 1)
        self.relative_decrease = kwargs.pop("relative_decrease", False)
        self.log_types = {"ntrial": "i", "alpha": "e"}
    def accept(self, state, step, solver, info):
        x = step.x
        alpha = 1.0

        for bt in range(self.maxiter):
            xtrial = solver.nd.add_update(x, alpha, step.dx)
            trial = solver.nd.evaluate(xtrial)

            if self.relative_decrease:
                meritvalue_aimed = (1-self.c * alpha )*state.meritvalue
            else:
                meritvalue_aimed = state.meritvalue + self.c * alpha * step.meritgrad
            # print(f"{meritvalue_aimed=} {trial.meritvalue=}")
            if trial.meritvalue <= meritvalue_aimed:
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
            failure="armijo backtracking failed",
        )

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