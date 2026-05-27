import numpy as np
import scipy.sparse.linalg as splinalg

from FEM2D.linalg.solver import IterativeSolver


#=================================================================#
class GeometricMultigrid(IterativeSolver):
    def __init__(self, As, transfers, **kwargs):
        self.method = "geommg"
        self.As = As
        self.transfers = transfers
        self.nu_pre = kwargs.pop("nu_pre", 2)
        self.nu_post = kwargs.pop("nu_post", 2)
        self.omega = kwargs.pop("omega", 0.8)

        super().__init__(kwargs)

        self._update_smoother_data()

    def _update_smoother_data(self):
        self.Dinv = [self.omega / A.diagonal() for A in self.As]

    def update(self, A=None, **kwargs):
        self._update_smoother_data()

    def _solve_impl(self, A=None, b=None, x0=None, maxiter=None, rtol=None, atol=None):
        if A is not None:
            self.As[-1] = A
            self._update_smoother_data()

        return self.solve_mg(
            b=b,
            x0=x0,
            maxiter=maxiter,
            rtol=rtol,
            atol=atol,
        )

    def solve_mg(self, b=None, x0=None, rtol=1e-8, atol=1e-14, maxiter=50):
        x = np.zeros_like(b) if x0 is None else x0.copy()

        normb = np.linalg.norm(b)
        threshold = max(atol, rtol * normb)

        for _ in range(maxiter):
            r = b - self.As[-1] @ x
            res = np.linalg.norm(r)

            self.callback(res)

            if res <= threshold:
                break

            x += self.vcycle(len(self.As) - 1, r)

        return x

    def smooth(self, ell, x, b, nu):
        A = self.As[ell]
        Dinv = self.Dinv[ell]

        for _ in range(nu):
            x += Dinv * (b - A @ x)

        return x

    def vcycle(self, ell, b):
        A = self.As[ell]

        if ell == 0:
            return splinalg.spsolve(A, b)

        x = np.zeros_like(b)

        x = self.smooth(ell, x, b, self.nu_pre)

        r = b - A @ x
        rc = self.transfers[ell - 1].restrict(r)

        ec = self.vcycle(ell - 1, rc)

        x += self.transfers[ell - 1].prolong(ec)

        x = self.smooth(ell, x, b, self.nu_post)

        return x