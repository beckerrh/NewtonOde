import numpy as np
from scipy import sparse
from types import SimpleNamespace
from Utility import timer


class CondensedLinearSolver:
    def __init__(self, homogeneous_dirichlet=True):
        self.homogeneous_dirichlet = homogeneous_dirichlet
        self.timer = timer.Timer()

    def condense(self, Ah, fh):
        Alocal = Ah.local
        Aglob = Ah.global_p1.copy().tocoo()

        nx, ncomp = fh.p1.shape
        ne = nx - 1

        nb = fh.bubbles.shape[1]
        nb_tot = nb * ncomp
        ndv = 2 * ncomp

        g = fh.p1.copy()

        if Alocal is None or nb_tot == 0:
            return SimpleNamespace(A=Aglob.tocsr(), b=g, bubble_solver_data=None)

        Apb = Alocal.Apb
        Abp = Alocal.Abp
        Abb = Alocal.Abb

        fb = fh.bubbles.reshape(ne, nb_tot)

        with self.timer("invert_Abb"):
            X_abp = np.linalg.solve(Abb, Abp)
            X_fb = np.linalg.solve(Abb, fb[..., None])[..., 0]

        with self.timer("schur"):
            Schur = - np.einsum("eib,ebj->eij", Apb, X_abp)
            rhs_vertex = np.einsum("eib,eb->ei", Apb, X_fb)

        g[:-1, :] -= rhs_vertex[:, :ncomp]
        g[1:, :] -= rhs_vertex[:, ncomp:]

        base = np.arange(ne) * ncomp
        elem_dofs = np.empty((ne, ndv), dtype=int)

        for c in range(ncomp):
            elem_dofs[:, c] = base + c
            elem_dofs[:, ncomp + c] = base + ncomp + c

        rows = np.broadcast_to(elem_dofs[:, :, None], (ne, ndv, ndv)).ravel()
        cols = np.broadcast_to(elem_dofs[:, None, :], (ne, ndv, ndv)).ravel()

        Slocal = sparse.coo_matrix(
            (Schur.ravel(), (rows, cols)),
            shape=Aglob.shape,
        )

        A = (Aglob + Slocal).tocsr()

        bubble_solver_data = SimpleNamespace(
            X_abp=X_abp,
            X_fb=X_fb,
            nb=nb,
            ncomp=ncomp,
        )

        return SimpleNamespace(A=A, b=g, bubble_solver_data=bubble_solver_data)


    def solve_global(self, A, b):
        sol = sparse.linalg.spsolve(A, b.ravel())
        return sol.reshape(b.shape)

    def recover_bubbles(self, bubble_solver_data, p1):
        if bubble_solver_data is None:
            return None

        X_abp = bubble_solver_data.X_abp
        X_fb = bubble_solver_data.X_fb
        nb = bubble_solver_data.nb
        ncomp = bubble_solver_data.ncomp

        p1loc = np.concatenate([p1[:-1, :], p1[1:, :]], axis=1)

        bubbles_flat = X_fb - np.einsum("ebj,ej->eb", X_abp, p1loc)

        return bubbles_flat.reshape(p1.shape[0] - 1, nb, ncomp)

    def solve(self, Ah, fh):
        if Ah.local is None or fh.bubbles.shape[1] == 0:
            with self.timer("global_solve"):
                p1 = self.solve_global(Ah.global_p1, fh.p1)
            return SimpleNamespace(
                p1=p1,
                bubbles=np.zeros_like(fh.bubbles),
            )

        with self.timer("condense"):
            condensed = self.condense(Ah, fh)

        with self.timer("global_solve"):
            p1 = self.solve_global(condensed.A, condensed.b)

        with self.timer("back_substitution"):
            bubbles = self.recover_bubbles(condensed.bubble_solver_data, p1)

        return SimpleNamespace(p1=p1, bubbles=bubbles)