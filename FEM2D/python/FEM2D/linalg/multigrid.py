import numpy as np
import scipy.sparse.linalg as splinalg

from Utility import timer
from .solver import IterativeSolver


#=================================================================#
class GeometricMultigrid(IterativeSolver):
    def __init__(self, As, transfers, **kwargs):
        self.method = "geommg"
        self.As = As
        self.transfers = transfers

        self.timer = kwargs.pop("timer", timer.Timer())


        self.nu_pre = kwargs.pop("nu_pre", 2)
        self.nu_post = kwargs.pop("nu_post", 2)
        self.omega = kwargs.pop("omega", 0.8)
        self.gamma = kwargs.pop("gamma", 0.99)
        self.coarse_threshold = kwargs.pop("coarse_threshold", 2000)

        self.smoother = kwargs.pop("smoother", "jacobi")
        self.smoother_kwargs = kwargs.pop("smoother_kwargs", {})

        self.acceleration = kwargs.pop("acceleration", "arnoldi")
        self.arnoldi_dim = kwargs.pop("arnoldi_dim", 4)
        self.arnoldi_breakdown = kwargs.pop("arnoldi_breakdown", 1e-13)

        super().__init__(kwargs)

        self._update_smoother_data()

    def _update_smoother_data(self):
        self.smoother_data = []

        for A in self.As:
            if self.smoother == "jacobi":
                data = self.omega / A.diagonal()

            elif self.smoother == "ilu":
                ilu = splinalg.spilu(A.tocsc(), **self.smoother_kwargs)
                data = ilu

            elif self.smoother == "none":
                data = None

            else:
                raise ValueError(f"unknown smoother {self.smoother!r}")

            self.smoother_data.append(data)
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

    def coarse_solve(self, ell, b):
        A = self.As[ell]

        if ell == 0 or A.shape[0] <= self.coarse_threshold:
            return splinalg.spsolve(A, b)

        return self.vcycle(ell, b)

    def arnoldi_acceleration(self, ell, r, m=None):
        """
        Right-preconditioned Arnoldi/GMRES correction.

        Approximately solves

            A e = r

        with right preconditioner B = one multigrid V-cycle:

            e in span{ B v_0, B v_1, ... }.

        Here:
            A = self.As[ell]
            B q = self.vcycle(ell, q)

        Returns
        -------
        e : ndarray
            Accelerated correction.
        """
        A = self.As[ell]

        if m is None:
            m = self.arnoldi_dim

        beta = np.linalg.norm(r)
        if beta == 0.0 or not np.isfinite(beta):
            return np.zeros_like(r)

        n = r.size

        V = np.zeros((m + 1, n), dtype=r.dtype)
        Z = np.zeros((m, n), dtype=r.dtype)
        H = np.zeros((m + 1, m), dtype=r.dtype)

        V[0] = r / beta

        g = np.zeros(m + 1, dtype=r.dtype)
        g[0] = beta

        k_done = 0

        for k in range(m):
            # right-preconditioned vector
            z = self.vcycle(ell, V[k])

            zn = np.linalg.norm(z)
            if zn == 0.0 or not np.isfinite(zn):
                break

            # normalize preconditioned direction to avoid huge Z columns
            z = z / zn
            Z[k] = z

            w = A @ z

            if not np.all(np.isfinite(w)):
                break

            # Modified Gram-Schmidt
            for j in range(k + 1):
                H[j, k] = np.dot(V[j], w)
                w -= H[j, k] * V[j]

            hn = np.linalg.norm(w)
            H[k + 1, k] = hn

            k_done = k + 1

            if hn < self.arnoldi_breakdown or not np.isfinite(hn):
                break

            V[k + 1] = w / hn

        if k_done == 0:
            return self.vcycle(ell, r)

        Hk = H[:k_done + 1, :k_done]
        gk = g[:k_done + 1]

        if not np.all(np.isfinite(Hk)):
            return self.vcycle(ell, r)

        hnorm = np.linalg.norm(Hk)
        if hnorm == 0.0 or not np.isfinite(hnorm):
            return self.vcycle(ell, r)

        try:
            y, *_ = np.linalg.lstsq(Hk / hnorm, gk / hnorm, rcond=None)
        except np.linalg.LinAlgError:
            return self.vcycle(ell, r)

        if not np.all(np.isfinite(y)):
            return self.vcycle(ell, r)

        Zk = Z[:k_done]

        if not np.all(np.isfinite(Zk)):
            return self.vcycle(ell, r)

        e = Zk.T @ y

        if not np.all(np.isfinite(e)):
            return self.vcycle(ell, r)

        return e
    def solve_mg(self, b=None, x0=None, rtol=1e-8, atol=1e-14, maxiter=50):
        x = np.zeros_like(b) if x0 is None else x0.copy()

        A = self.As[-1]
        normb = np.linalg.norm(b)
        threshold = max(atol, rtol * normb)

        for _ in range(maxiter):
            r = b - A @ x
            res = np.linalg.norm(r)

            self.callback(res)

            if res <= threshold:
                break

            if self.acceleration == "arnoldi":
                with self.timer("arnoldi"):
                    dx = self.arnoldi_acceleration(len(self.As) - 1, r)
            else:
                dx = self.vcycle(len(self.As) - 1, r)

            x += self.gamma * dx

        return x

    def smooth(self, ell, x, b, nu):
        A = self.As[ell]
        data = self.smoother_data[ell]

        if self.smoother == "none":
            return x

        if self.smoother == "jacobi":
            Dinv = data
            for _ in range(nu):
                x += Dinv * (b - A @ x)
            return x

        if self.smoother == "ilu":
            ilu = data
            for _ in range(nu):
                r = b - A @ x
                x += ilu.solve(r)
            return x

        raise ValueError(f"unknown smoother {self.smoother!r}")
    def vcycle(self, ell, b):
        A = self.As[ell]

        if ell == 0:
            return splinalg.spsolve(A, b)

        x = np.zeros_like(b)
        x = self.smooth(ell, x, b, self.nu_pre)
        r = b - A @ x
        rc = self.transfers[ell - 1].restrict(r)
        ec = self.coarse_solve(ell - 1, rc)
        x += self.gamma * self.transfers[ell - 1].prolong(ec)
        x = self.smooth(ell, x, b, self.nu_post)

        return x