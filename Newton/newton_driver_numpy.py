from types import SimpleNamespace
import numpy as np

#=========================================================
class NewtonDriverNumpy:
    def __init__(self, **kwargs):
        if 'test_problem' in kwargs:
            test_problem = kwargs['test_problem']
            self.F = test_problem.F
            self.dF = test_problem.dF
        else:
            self.F = kwargs.pop('F')
            self.dF = kwargs.pop('dF')
    def add_update(self, x, alpha, dx):
        return x + alpha * dx
    def evaluate(self, x):
        xnorm = np.linalg.norm(x)
        r = self.F(x)
        meritvalue = 0.5*np.sum(r**2)
        return SimpleNamespace(residual=r, meritvalue=meritvalue, norm_X=xnorm)
    def compute_newton_step(self,  x, state, iterdata):
        r = state.residual
        J = np.atleast_2d(self.dF(x).squeeze())
        r = np.atleast_1d(r.squeeze())
        try:
            dx = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError as e:
            print(f"{x=}")
            print(f"{J=}")
            print(f"{r=}")
            raise e
        meritgrad = -np.sum(r ** 2)
        return SimpleNamespace(x=x, dx=dx, dx_norm=np.linalg.norm(dx), meritgrad=meritgrad)

    def compute_regularized_newton_step(self, x, state, iterdata, lambda_reg):
        r = np.atleast_1d(state.residual).reshape(-1)

        J0 = np.atleast_2d(self.dF(x).squeeze())
        n = r.shape[0]
        Jreg = J0 + lambda_reg * np.eye(n)

        dx = np.linalg.solve(Jreg, -r)

        rlin = r + J0 @ dx
        predicted_reduction = 0.5 * np.sum(r ** 2) - 0.5 * np.sum(rlin ** 2)

        meritgrad = float(r @ (J0 @ dx))

        return SimpleNamespace(
            x=x,
            dx=dx,
            dx_norm=np.linalg.norm(dx),
            meritgrad=meritgrad,
            predicted_reduction=predicted_reduction,
            lambda_reg=lambda_reg,
            liniter=0,
        )
    # def compute_regularized_newton_step(self, x, state, info, lambda_reg=0.0):
    #     F = np.atleast_1d(state.residual).reshape(-1)
    #     J = np.atleast_2d(self.dF(x).squeeze())
    #
    #     g_merit = J.T @ F
    #
    #     # crude switch for now
    #     theta = 0.0 if lambda_reg < 100.0 else 1.0
    #     g_rhs = theta * g_merit
    #
    #     A = J + lambda_reg * np.eye(J.shape[0])
    #     b = -F - lambda_reg * g_rhs
    #
    #     try:
    #         dx = np.linalg.solve(A, b)
    #     except np.linalg.LinAlgError:
    #         dx = -g_merit
    #
    #     Jdx = J @ dx
    #
    #     predicted_reduction = -(
    #             np.dot(g_merit, dx) + 0.5 * np.dot(Jdx, Jdx)
    #     )
    #
    #     return SimpleNamespace(
    #         dx=dx,
    #         dx_norm=np.linalg.norm(dx),
    #         update_norm=np.linalg.norm(dx),
    #         liniter=0,
    #         meritgrad=np.dot(g_merit, dx),
    #         predicted_reduction=predicted_reduction,
    #         lambda_reg=lambda_reg,
    #     )
    def compute_gradient_step(self, x, state, iterdata):
        r = np.atleast_1d(state.residual).reshape(-1)
        J = np.atleast_2d(self.dF(x).squeeze())

        dx = -(J.T @ r).reshape(-1)

        return SimpleNamespace(
            x=x,
            dx=dx,
            dx_norm=np.linalg.norm(dx),
            meritgrad=-np.sum(dx ** 2),
            liniter=0,
            step_type="gradient",
        )
    def merit_directional_derivative(self, x, state, dx):
        r = np.atleast_1d(state.residual).reshape(-1)
        J = np.atleast_2d(self.dF(x).squeeze())
        dx = np.atleast_1d(dx).reshape(-1)
        return float(r @ (J @ dx))
