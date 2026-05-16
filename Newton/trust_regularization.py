from types import SimpleNamespace
import numpy as np
import armijo

# ======================================================================
class TrustRegularizationGlobalization:
    def __init__(self, **kwargs):
        self.rho_good = kwargs.pop("rho_good", 0.75)
        self.rho_bad = kwargs.pop("rho_bad", 0.2)
        self.rho_accept = kwargs.pop("rho_accept", 0.1)
        self.lambda_reg = kwargs.pop("lambda_reg", 1e-8)
        self.decrease = kwargs.pop("decrease", 0.25)
        self.increase = kwargs.pop("increase", 4.0)
        self.maxiter = kwargs.pop("maxiter", 10)
        self.lambda_max = kwargs.pop("lambda_max", 1e6)
        self.lambda_min = kwargs.pop("lambda_min", 1e-8)
        self.lambda_reset = kwargs.pop("lambda_reset", 1.0)
        self.fallback = kwargs.pop(
            "fallback",
            armijo.ArmijoGlobalization(maxiter=20, omega=0.5, c=1e-4)
        )
        self.verbose = kwargs.pop("verbose", 1)
        self.log_types = {
            "ntrial": "i",
            "lambda_reg": "e",
            "rho": "f",
        }
    def accept(self, state, step, solver, info):
        x = step.x
        lam = self.lambda_reg

        last_step = None
        last_rho = np.nan
        last_actual = np.nan
        last_predicted = np.nan
        last_lambda = lam

        for ntrial in range(1, self.maxiter + 1):
            last_lambda = lam

            request = SimpleNamespace(
                step_type="regularized_newton",
                lambda_reg=lam,
            )

            step = solver.build_step(x, state, request=request)

            xtrial = solver.nd.add_update(x, 1.0, step.dx)
            trial = solver.nd.evaluate(xtrial)

            actual = state.meritvalue - trial.meritvalue
            predicted = getattr(step, "predicted_reduction", np.nan)
            rho = actual / predicted if predicted > 0 else -1.0

            last_step = step
            last_actual = actual
            last_predicted = predicted
            last_rho = rho

            success = (rho >= self.rho_accept) and (actual > 0.0)

            if predicted <= 0.0:
                if step.meritgrad > 0.0:
                    rev_step = SimpleNamespace(**step.__dict__)
                    rev_step.dx = -step.dx
                    rev_step.meritgrad = -step.meritgrad
                    rev_step.step_type = "rev-reg-newt"

                    acc = self.fallback.accept(
                        state=state,
                        step=rev_step,
                        solver=solver,
                        info=info,
                    )

                    if acc.success:
                        self.lambda_reg = self.lambda_reset
                        acc.step = rev_step
                        acc.lambda_reg = lam
                        acc.lambda_next = self.lambda_reg
                        acc.rho = np.nan
                        acc.ntrial = ntrial
                        return acc

                lam *= self.increase
                if lam > self.lambda_max:
                    break
                continue
            # if self.verbose:
            #     print(
            #         f"TR trial {ntrial:2d}: "
            #         f"lambda={lam:.3e} "
            #         f"|dx|={step.dx_norm:.3e} "
            #         f"phi={trial.meritvalue:.3e} "
            #         f"actual={actual:.3e} "
            #         f"pred={predicted:.3e} "
            #         f"rho={rho:.3e} "
            #         f"accept={rho >= self.rho_accept and actual > 0.0}"
            #     )

            if success:
                if rho > self.rho_good:
                    self.lambda_reg = max(lam * self.decrease, self.lambda_min)
                elif rho < self.rho_bad:
                    self.lambda_reg = min(lam * self.increase, self.lambda_max)
                else:
                    self.lambda_reg = lam

                return SimpleNamespace(
                    success=True,
                    x=xtrial,
                    state=trial,
                    step=step,
                    alpha=1.0,
                    ntrial=ntrial,
                    rho=rho,
                    actual=actual,
                    predicted=predicted,
                    lambda_reg=lam,
                    lambda_next=self.lambda_reg,
                    failure=None,
                )

            lam *= self.increase

            if lam > self.lambda_max:
                break

        # self.lambda_reg = lam
        lambda_failed = lam

        if self.verbose:
            print(
                "TR fallback: "
                f"last lambda={last_lambda:.3e}, "
                f"next lambda={lam:.3e}, "
                f"last rho={last_rho:.3e}, "
                f"actual={last_actual:.3e}, "
                f"predicted={last_predicted:.3e}, "
                f"|dx|={getattr(last_step, 'dx_norm', np.nan):.3e}"
            )
        if hasattr(solver.nd, "compute_gradient_step"):
            gstep = solver.build_step(
                x, state,
                request=SimpleNamespace(step_type="gradient")
            )

            acc = self.fallback.accept(
                state=state,
                step=gstep,
                solver=solver,
                info=info,
            )

            acc.step = gstep
            acc.lambda_reg = last_lambda
            acc.lambda_next = lam
            acc.rho = np.nan
            acc.ntrial = ntrial

            if not acc.success:
                acc.failure = (
                    f"regularized Newton rejected and gradient fallback failed; "
                    f"last lambda={last_lambda:.3e}, "
                    f"last rho={last_rho:.3e}, "
                    f"last actual={last_actual:.3e}, "
                    f"last predicted={last_predicted:.3e}"
                )
            if acc.success:
                self.lambda_reg = self.lambda_reset
                return acc
            else:
                self.lambda_reg = lambda_failed
            return acc
        return SimpleNamespace(
            success=False,
            x=x,
            state=state,
            step=last_step,
            alpha=0.0,
            ntrial=self.maxiter,
            rho=last_rho,
            actual=last_actual,
            predicted=last_predicted,
            lambda_reg=last_lambda,
            lambda_next=self.lambda_reg,
            failure=(
                f"regularized Newton rejected after {self.maxiter} trials; "
                f"last lambda={last_lambda:.3e}, "
                f"last rho={last_rho:.3e}, "
                f"last actual={last_actual:.3e}, "
                f"last predicted={last_predicted:.3e}"
            )
        )