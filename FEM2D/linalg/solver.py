import numpy as np
import Utility

#=================================================================#
class IterativeSolver:
    def __repr__(self):
        return f"{self.method}_{self.maxiter}_{self.rtol}"

    def __init__(self, kwargs):
        self.scale = kwargs.pop("scale", False)
        self.atol = kwargs.pop("atol", 1e-14)
        self.rtol = kwargs.pop("rtol", 1e-8)
        self.maxiter = kwargs.pop("maxiter", 100)
        self.disp = kwargs.pop("disp", 0)

        self.niter = 0
        self.residuals = []

        self.logger = kwargs.pop("logger", None)
        if self.logger is None:
            types = {"it": "i", "res": "e"}
            self.logger = Utility.logger.Logger(
                verbose=self.disp,
                types=types,
                name=self.__class__.__name__,
            )

    def callback(self, rk=None):
        self.niter += 1

        data = {"it": self.niter}

        if rk is not None:
            res = abs(rk) if np.isscalar(rk) else np.linalg.norm(rk)
            self.residuals.append(res)
            data["res"] = res

        if self.disp:
            self.logger.update(data)

    def solve(self, A=None, b=None, maxiter=None, rtol=None, atol=None, x0=None):
        if maxiter is None:
            maxiter = self.maxiter
        if rtol is None:
            rtol = self.rtol
        if atol is None:
            atol = self.atol

        self.niter = 0
        self.residuals.clear()

        bsolve = b
        if A is not None and self.scale and hasattr(A, "scale_matrix"):
            bsolve = b.copy()
            A.scale_vec(bsolve)

        sol = self._solve_impl(
            A=A,
            b=bsolve,
            x0=x0,
            maxiter=maxiter,
            rtol=rtol,
            atol=atol,
        )

        if A is not None and self.scale and hasattr(A, "scale_matrix"):
            A.scale_vec(sol)

        return sol

    def _solve_impl(self, A=None, b=None, x0=None, maxiter=None, rtol=None, atol=None):
        raise NotImplementedError