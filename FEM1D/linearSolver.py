import numpy as np
from scipy import sparse
from types import SimpleNamespace

class CondensedLinearSolver:
    def __init__(self, homogeneous_dirichlet=True):
        self.homogeneous_dirichlet = homogeneous_dirichlet


    # OLD, no longer needed, is done before !!
    def apply_dirichlet_p1(self, A, b, ncomp):
        return A,b
        nx = b.shape[0]

        bc = []
        for c in range(ncomp):
            bc.append(c)
            bc.append((nx - 1) * ncomp + c)

        A = A.tolil()
        b = b.copy()

        for i in bc:
            A[i, :] = 0.0
            A[i, i] = 1.0

        b[0, :] = 0.0
        b[-1, :] = 0.0

        return A.tocsc(), b

    def solve(self, Ah, fh):
        Alocal, Ap1 = Ah.local, Ah.global_p1

        nx, ncomp = fh.p1.shape

        if Alocal is None:
            # A, b = self.apply_dirichlet_p1(Ap1, fh.p1, ncomp)
            sol = sparse.linalg.spsolve(A, b.ravel())
            return SimpleNamespace(p1=sol.reshape(fh.p1.shape), bubbles=None)

        ne = nx - 1
        nb = fh.bubbles.shape[1]
        nb_tot = nb * ncomp

        Sglob = Ap1.copy().tolil()
        g = fh.p1.copy()

        bubble_solver_data = []

        for e in range(ne):
            Apb = Alocal.Apb[e]
            Abp = Alocal.Abp[e]
            Abb = Alocal.Abb[e]

            fb = fh.bubbles[e].reshape(nb_tot)

            X = np.linalg.solve(Abb, Abp)
            y = np.linalg.solve(Abb, fb)

            Se_corr = Apb @ X
            ge_corr = Apb @ y

            gdofs = []
            for node in [e, e + 1]:
                for c in range(ncomp):
                    gdofs.append(node * ncomp + c)

            Sglob[np.ix_(gdofs, gdofs)] -= Se_corr

            ge_corr = ge_corr.reshape(2, ncomp)
            g[e, :]     -= ge_corr[0, :]
            g[e + 1, :] -= ge_corr[1, :]

            bubble_solver_data.append((Abb, Abp, fb))

        # Sglob, g = self.apply_dirichlet_p1(Sglob, g, ncomp)

        sol = sparse.linalg.spsolve(Sglob, g.ravel())
        p1 = sol.reshape(nx, ncomp)

        bubbles = np.zeros_like(fh.bubbles)

        for e in range(ne):
            Abb, Abp, fb = bubble_solver_data[e]
            up = np.concatenate([p1[e, :], p1[e + 1, :]])
            ub = np.linalg.solve(Abb, fb - Abp @ up)
            bubbles[e, :, :] = ub.reshape(nb, ncomp)

        return SimpleNamespace(p1=p1, bubbles=bubbles)