import numpy as np
from scipy import sparse
from types import SimpleNamespace
import bernstein, linearSolver
from Utility import mesh1d, plotting, logger, timer

"""
alpha = diffusion_coef
beta = convection_coef

-(alpha u')' + beta u'  = f(u)

tangent equation at given u (only f depends on u):

-(alpha p')' + beta p' - f'_u p = f(u) +(alpha u')' - beta u'

for the linearized estimator we use
-(alpha (u+p)')' + beta (u+p)' - f'_u p = f

"""


#-------------------------------------------------------------
class LinearElliptic:
    # -------------------------------------------------------------
    def __init__(self, korder, app):
        self.korder, self.app = korder, app
        self.integration = {'coeff': bernstein.reference_integration(2 * korder + 2, korder),
                            'rhs': bernstein.reference_integration(2*korder + 2, korder),
                            # 'est': bernstein.reference_integration(2*korder + 2, korder),
                            'error': bernstein.reference_integration(2*korder + 3, korder)}
        self.linear_solver = linearSolver.CondensedLinearSolver()
        self.timer = timer.Timer()

    # -------------------------------------------------------------
    # ------------           helpers            -------------------
    # -------------------------------------------------------------

    def is_scalar_coefficient(self, C, xq):
        C = np.asarray(C)
        return C.shape == xq.shape or C.shape == () or C.ndim == 0

    def normalize_coefficient(self, A, xq):
        """
        Normalize coefficient shapes.

        scalar constant:
            () -> (ne,nq)

        scalar field:
            (ne,nq) unchanged

        constant matrix:
            (c,d) -> (ne,nq,c,d)

        matrix field:
            (ne,nq,c,d) unchanged
        """
        A = np.asarray(A)

        # scalar constant
        if A.ndim == 0:
            return A * np.ones_like(xq)

        # scalar field
        if A.shape == xq.shape:
            return A

        # constant matrix
        if A.ndim == 2:
            return np.broadcast_to(A, xq.shape + A.shape)

        # already matrix-valued field
        return A

    def apply_coefficient(self, A, v):
        """
        Apply scalar or matrix coefficient A to vector field v.

        v shape:
            (..., ncomp)

        returns:
            (..., ncomp)
        """
        A = np.asarray(A)

        # scalar coefficient
        if A.ndim == v.ndim - 1:
            return A[..., None] * v

        # matrix coefficient
        return np.einsum("...ab,...b->...a", A, v)
    def plot_basis(self):
        import matplotlib.pyplot as plt
        x = np.linspace(0, 1)
        p = bernstein.bernstein_basis(self.korder, x)
        # print(f"{p.shape=}")
        plt.figure(figsize=(12, 4))
        for i in range(p.shape[0]):
            plt.plot(x, p[i], '-', label=fr"$\phi_{{{i}}}$")
        plt.legend()
        plt.grid()
        plt.show()
    def add_update(self, u, alpha, p):
        bubbles = None if u.bubbles is None else u.bubbles + alpha * p.bubbles
        return SimpleNamespace(
            p1=u.p1 + alpha * p.p1,
            bubbles=bubbles
        )
    def make_fe_vector(self, mesh):
        nx, k, ncomp = mesh.shape[0], self.korder, self.app.ncomp
        bubbles = np.zeros(shape=(nx - 1, k - 1, ncomp)) if k > 1 else None
        return SimpleNamespace(p1=np.zeros(shape=(nx, ncomp)), bubbles=bubbles)
    def make_local_fe_matrix(self, mesh):
        nx, k, ncomp = mesh.shape[0], self.korder, self.app.ncomp
        nb = max(k - 1, 0) * ncomp
        nv = 2 * ncomp
        return SimpleNamespace(
            Apb=np.zeros(shape=(nx - 1, nv, nb)) if k > 1 else None,
            Abp=np.zeros(shape=(nx - 1, nb, nv)) if k > 1 else None,
            Abb=np.zeros(shape=(nx - 1, nb, nb)) if k > 1 else None,
        )
    def local_coeffs(self, mesh, uh, ph=None):
        ne, k, ncomp = mesh.shape[0] - 1, self.korder, self.app.ncomp
        uloc = np.zeros((ne, k + 1, ncomp))
        uloc[:, 0, :] = uh.p1[:-1, :]
        uloc[:, k, :] = uh.p1[1:, :]
        if k > 1:
            uloc[:, 1:k, :] = uh.bubbles
        if ph is not None:
            uloc[:, 0, :] += ph.p1[:-1, :]
            uloc[:, k, :] += ph.p1[1:, :]
            if k > 1:
                uloc[:, 1:k, :] += ph.bubbles
        return uloc
    def eval_fe_on_grid(self, mesh, uh, nsub=20):
        k = self.korder
        uloc = self.local_coeffs(mesh, uh)

        xi = np.linspace(0.0, 1.0, nsub)
        phi = bernstein.bernstein_basis(k, xi)

        xs_all = []
        us_all = []

        for e in range(len(mesh) - 1):
            xL, xR = mesh[e], mesh[e + 1]

            xs = xL + (xR - xL) * xi
            us = np.einsum("jc,jq->qc", uloc[e], phi)

            xs_all.append(xs)
            us_all.append(us)

        return np.concatenate(xs_all), np.concatenate(us_all, axis=0)
    def eval_fe_at_points(self, mesh, uh, xpts):
        k = self.korder
        ncomp = self.app.ncomp
        uloc = self.local_coeffs(mesh, uh)

        out = np.empty((len(xpts), ncomp))

        for i, x in enumerate(xpts):
            e = np.searchsorted(mesh, x, side="right") - 1
            e = min(max(e, 0), len(mesh) - 2)

            xi = (x - mesh[e]) / (mesh[e + 1] - mesh[e])
            phi = bernstein.bernstein_basis(k, np.array([xi]))[:, 0]

            out[i, :] = phi @ uloc[e]

        return out
    def expand_scalar_local_to_system(self, A_s):
        ne, nk, _ = A_s.shape
        ncomp = self.app.ncomp
        I = np.eye(ncomp)
        return np.einsum(
            "eij,ab->eiajb",
            A_s, I,
            optimize=True,
        ).reshape(ne, nk * ncomp, nk * ncomp)
    def normalize_jacobian(self, J, xq, u_vals):
        ne, nq = xq.shape
        ncomp = self.app.ncomp

        J = np.asarray(J)

        if J.shape == (ne, nq, ncomp, ncomp):
            return J

        if ncomp == 1:
            if J.shape == (ne, nq):
                return J[:, :, None, None]
            if J.shape == (ne, nq, 1):
                return J[:, :, :, None]

        raise ValueError(
            f"df_du has shape {J.shape}, expected {(ne, nq, ncomp, ncomp)}"
        )
    # -------------------------------------------------------------
    def interpolate_to_new_mesh(self, mesh_old, uh_old, mesh_new, maps):
        k = self.korder
        uh_new = self.make_fe_vector(mesh_new)

        refined_map, non_refined_map = maps

        uloc_old = self.local_coeffs(mesh_old, uh_old)
        ne_new = mesh_new.shape[0] - 1
        uloc_new = np.zeros((ne_new, k + 1, self.app.ncomp))

        # unchanged old elements
        for old_e, new_e in non_refined_map:
            uloc_new[new_e, :, :] = uloc_old[old_e, :, :]

        # bisected old elements
        for old_e, new_left_e, new_right_e in refined_map:
            cleft, cright = bernstein.subdivide_bernstein_midpoint(
                uloc_old[old_e, :, :]
            )
            uloc_new[new_left_e, :, :] = cleft
            uloc_new[new_right_e, :, :] = cright

        # local Bernstein coefficients -> global FE vector
        uh_new.p1[:-1, :] = uloc_new[:, 0, :]
        uh_new.p1[-1, :] = uloc_new[-1, -1, :]

        if uh_new.bubbles is not None: uh_new.bubbles[:, :, :] = uloc_new[:, 1:k, :]

        return uh_new
    def test_interpolation(self, mesh):
        k = self.korder
        uh = self.make_fe_vector(mesh)

        def ufun(x):
            return x**self.korder
            # return np.sin(7 * x)  # or x**k, x**2, etc.

        # set local Bernstein coefficients by interpolation at Bernstein nodes
        xi_nodes = np.linspace(0, 1, k + 1)
        B = bernstein.bernstein_basis(k, xi_nodes).T

        for e in range(len(mesh) - 1):
            xloc = mesh[e] + (mesh[e + 1] - mesh[e]) * xi_nodes
            vals = ufun(xloc)[:, None]
            c = np.linalg.solve(B, vals)

            uh.p1[e, :] = c[0, :]
            uh.p1[e + 1, :] = c[-1, :]
            if k > 1:
                uh.bubbles[e, :, :] = c[1:-1, :]

        eta_sq = np.ones(len(mesh) - 1)
        mesh_new, maps = mesh1d.adapt_mesh(mesh, eta_sq, theta=0.75)
        uh_new = self.interpolate_to_new_mesh(mesh, uh, mesh_new, maps)

        x_old, u_old = self.eval_fe_on_grid(mesh, uh, nsub=20)
        x_new, u_new = self.eval_fe_on_grid(mesh_new, uh_new, nsub=20)
        x_new, u_new = self.eval_fe_on_grid(mesh_new, uh_new, nsub=20)

        # evaluate old FE function at x_new
        u_old_at_xnew = self.eval_fe_at_points(mesh, uh, x_new)

        err = np.max(np.abs(u_new[:, 0] - u_old_at_xnew[:, 0]))
        print("prolongation error =", err)
        err = np.max(np.abs(np.interp(x_new, x_old, u_old[:, 0]) - u_new[:, 0]))
        print("interpolation max error =", err)

    # -------------------------------------------------------------
    def make_laplace_matrix(self, mesh):
        nx, k, ncomp = mesh.shape[0], self.korder, self.app.ncomp
        ne = nx - 1

        Alocal = None if k == 1 else self.make_local_fe_matrix(mesh)
        Aglob = sparse.lil_matrix((nx * ncomp, nx * ncomp))

        A_lap_s = self.make_laplace_local_scalar(mesh)
        A = self.expand_scalar_local_to_system(A_lap_s)

        iv = []
        for node in [0, k]:
            for c in range(ncomp):
                iv.append(node * ncomp + c)

        ib = []
        for node in range(1, k):
            for c in range(ncomp):
                ib.append(node * ncomp + c)

        App = A[:, iv, :][:, :, iv]

        for e in range(ne):
            vertices = [e, e + 1]

            for a_node_loc, a_node in enumerate(vertices):
                for b_node_loc, b_node in enumerate(vertices):
                    block = App[
                            e,
                            a_node_loc * ncomp:(a_node_loc + 1) * ncomp,
                            b_node_loc * ncomp:(b_node_loc + 1) * ncomp,
                            ]

                    rows = slice(a_node * ncomp, (a_node + 1) * ncomp)
                    cols = slice(b_node * ncomp, (b_node + 1) * ncomp)

                    Aglob[rows, cols] += block

        if k > 1:
            Alocal.Apb[:, :, :] = A[:, iv, :][:, :, ib]
            Alocal.Abp[:, :, :] = A[:, ib, :][:, :, iv]
            Alocal.Abb[:, :, :] = A[:, ib, :][:, :, ib]

        Ah = SimpleNamespace(local=Alocal, global_p1=Aglob.tocsr())
        return self.dirichlet_matrix(mesh, Ah)
    # -------------------------------------------------------------
    def make_diffusion_local_scalar(self, mesh):
        integ = self.integration["coeff"]
        dx = mesh[1:] - mesh[:-1]
        xm = mesh[:-1]

        xq = xm[:, None] + dx[:, None] * integ.xi[None, :]
        alpha = self.app.diffusion_coef(xq)  # (ne, nq)

        return np.einsum(
            "eq,iq,jq,q,e->eij",
            alpha,
            integ.dphi,
            integ.dphi,
            integ.w,
            1.0 / dx,
            optimize=True,
        )
    def make_diffusion_local_matrix(self, mesh):
        app = self.app
        integ = self.integration["coeff"]

        x0 = mesh[:-1]
        dx = mesh[1:] - mesh[:-1]
        xq = x0[:, None] + dx[:, None] * integ.xi[None, :]

        A = app.diffusion_coef(xq)
        A = np.asarray(A)

        # allow constant matrix A.shape == (c,d)
        if A.ndim == 2:
            A = np.broadcast_to(A, xq.shape + A.shape)

        # A:     (ne,nq,c,d)
        # dphi:  (k+1,nq)
        # w:     (nq,)
        # dx scaling: derivative gives 1/dx^2, integral gives dx, so factor 1/dx
        Aloc = np.einsum(
            "eqcd,iq,jq,q,e->eicjd",
            A,
            integ.dphi,
            integ.dphi,
            integ.w,
            1.0 / dx,
            optimize=True,
        )
        ne, ni, c, nj, d = Aloc.shape
        return Aloc.reshape(ne, ni * c, nj * d)
    def make_laplace_local_scalar(self, mesh):
        integ = self.integration["coeff"]
        dx = mesh[1:] - mesh[:-1]
        Aref = np.einsum("iq,jq,q->ij", integ.dphi, integ.dphi, integ.w)
        return np.einsum("e,ij->eij", 1.0 / dx, Aref)
    # def matrix_p1_fast(self, mesh):
    #     nx, ncomp = mesh.shape[0], self.app.ncomp
    #     h = mesh[1:] - mesh[:-1]
    #     main = np.zeros(nx)
    #     off = np.zeros(nx - 1)
    #     main[:-1] += 1.0 / h
    #     main[1:] += 1.0 / h
    #     off[:] = -1.0 / h
    #     A1 = sparse.diags(
    #         diagonals=[off, main, off],
    #         offsets=[-1, 0, 1],
    #         shape=(nx, nx),
    #         format="csc"
    #     )
    #     if ncomp == 1:
    #         return A1
    #     return sparse.kron(A1, sparse.eye(ncomp, format="csc"), format="csc")
    # -------------------------------------------------------------
    def normH1(self, mesh, uh):
        integ = self.integration['coeff']
        xm, dx = mesh[:-1],  (mesh[1:] - mesh[:-1])
        uloc = self.local_coeffs(mesh, uh)
        du_ref = np.einsum('ijc,jq->iqc', uloc, integ.dphi)
        du_dx = du_ref / dx[:, None, None]
        return  np.sqrt(np.einsum('iqc, i, q -> ', du_dx**2, dx, integ.w))
    def normHminus1(self, mesh, rh):
        Ah_lap = self.make_laplace_matrix(mesh)
        # zh = self.solve_linear(mesh, Ah_lap, rh)
        zh = self.linear_solver.solve(Ah_lap, rh)
        val = np.sum(rh.p1 * zh.p1)
        if rh.bubbles is not None:
            val += np.sum(rh.bubbles * zh.bubbles)
        return np.sqrt(val)

    # -------------------------------------------------------------
    def compute_error(self, mesh, uh):
        ne, k, ncomp = mesh.shape[0] - 1, self.korder, self.app.ncomp
        integ = self.integration['error']
        xm, dx = mesh[:-1],  (mesh[1:] - mesh[:-1])
        x_rhs = xm[:, None] + dx[:, None] * integ.xi[None, :]  # shape (nx-1, int_n)
        uloc = self.local_coeffs(mesh, uh)
        u_vals = np.einsum('ijc,jq->iqc', uloc, integ.phi)
        du_ref = np.einsum('ijc,jq->iqc', uloc, integ.dphi)
        du_dx = du_ref / dx[:, None, None]
        u_ex = self.app.solution(x_rhs)
        if ncomp == 1 and u_ex.ndim == 2:
            u_ex = u_ex[:, :, None]
        du_ex = self.app.dsolution(x_rhs)
        if self.app.ncomp == 1 and du_ex.ndim == 2:
            du_ex = du_ex[:, :, None]
        eL2 = np.einsum('iqc, i, q -> ', (u_vals -u_ex)**2, dx, integ.w)
        eH1 = np.einsum('iqc, i, q -> ', (du_dx - du_ex)**2, dx, integ.w)
        return np.sqrt(eL2), np.sqrt(eH1)
    # -------------------------------------------------------------
    def compute_estimator_linearized(self, mesh, uh, ph):
        ne, k, ncomp = mesh.shape[0] - 1, self.korder, self.app.ncomp
        integ = self.integration['error']
        xm, dx = mesh[:-1],  (mesh[1:] - mesh[:-1])
        x_rhs = xm[:, None] + dx[:, None] * integ.xi[None, :]  # shape (nx-1, int_n)
        uloc = self.local_coeffs(mesh, uh)
        uploc = self.local_coeffs(mesh, uh, ph)
        # print(f"{uloc.shape=} {integ.phi.shape=}")
        u_vals = np.einsum('ijc,jq->iqc', uloc, integ.phi)
        up_vals = np.einsum('ijc,jq->iqc', uploc, integ.phi)
        p_vals = up_vals - u_vals

        dup_ref = np.einsum('ijc,jq->iqc', uploc, integ.dphi)
        d2up_ref = np.einsum('ijc,jq->iqc', uploc, integ.d2phi)
        dup_dx = dup_ref / dx[:, None, None]
        d2up_dx2 = d2up_ref / dx[:, None, None] ** 2
        f_all = self.app.f(x_rhs, u_vals)
        if ncomp == 1 and f_all.ndim == 2:
            f_all = f_all[:, :, None]

        df_du_all = self.app.df_du(x_rhs, u_vals)
        df_du_all = self.normalize_jacobian(df_du_all, x_rhs, u_vals)
        # expected shape: (ne, nq, ncomp, ncomp)

        Jp = np.einsum("iqab,iqb->iqa", df_du_all, p_vals)

        alpha = self.normalize_coefficient(self.app.diffusion_coef(x_rhs), x_rhs)
        alpha_x = self.normalize_coefficient(self.app.diffusion_coef_x(x_rhs), x_rhs)
        res = (
                f_all
                + self.apply_coefficient(alpha, d2up_dx2)
                + self.apply_coefficient(alpha_x, dup_dx)
                + Jp
        )
        # beta = self.normalize_coefficient(self.app.convection_coef(x_rhs), x_rhs)
        # res -= self.apply_coefficient(beta, dup_dx)
        if k>=2:
            # Build projection basis once, e.g. Bernstein degree k-2
            psi = bernstein.bernstein_basis(k - 2, integ.xi)  # (k-1, nq)
            M = np.einsum('aq,bq,q->ab', psi, psi, integ.w)
            rhs_proj = np.einsum('iqc,aq,q->iac', res, psi, integ.w)
            coef = np.linalg.solve(M, rhs_proj.reshape(ne, k - 1, ncomp))
            Pi_res = np.einsum('iac,aq->iqc', coef, psi)
            res -= Pi_res
        eta_vol_sq = np.einsum(
            'iqc,i,q->i',
            res ** 2,
            dx ** 3,
            integ.w
        )
        return SimpleNamespace(deta_global=np.sqrt(np.sum(eta_vol_sq))/np.pi/k, deta=eta_vol_sq)
    # -------------------------------------------------------------
    def compute_estimator_nonlinear(self, mesh, uh):
        ne, k, ncomp = mesh.shape[0] - 1, self.korder, self.app.ncomp
        integ = self.integration['error']
        xm, dx = mesh[:-1],  (mesh[1:] - mesh[:-1])
        x_rhs = xm[:, None] + dx[:, None] * integ.xi[None, :]  # shape (nx-1, int_n)
        uloc = self.local_coeffs(mesh, uh)
        # print(f"{uloc.shape=} {integ.phi.shape=}")
        u_vals = np.einsum('ijc,jq->iqc', uloc, integ.phi)
        du_ref = np.einsum('ijc,jq->iqc', uloc, integ.dphi)
        d2u_ref = np.einsum('ijc,jq->iqc', uloc, integ.d2phi)
        du_dx = du_ref / dx[:, None, None]
        d2u_dx2 = d2u_ref / dx[:, None, None] ** 2
        f_all = self.app.f(x_rhs, u_vals)
        if ncomp == 1 and f_all.ndim == 2:
            f_all = f_all[:, :, None]

        alpha = self.normalize_coefficient(self.app.diffusion_coef(x_rhs), x_rhs)
        alpha_x = self.normalize_coefficient(self.app.diffusion_coef_x(x_rhs), x_rhs)

        res = (
                f_all
                + self.apply_coefficient(alpha, d2u_dx2)
                + self.apply_coefficient(alpha_x, du_dx)
        )
        # beta = self.normalize_coefficient(self.app.convection_coef(x_rhs), x_rhs)
        # res -= self.apply_coefficient(beta, du_dx)
        if k>=2:
            # Build projection basis once, e.g. Bernstein degree k-2
            psi = bernstein.bernstein_basis(k - 2, integ.xi)  # (k-1, nq)
            M = np.einsum('aq,bq,q->ab', psi, psi, integ.w)
            rhs_proj = np.einsum('iqc,aq,q->iac', res, psi, integ.w)
            coef = np.linalg.solve(M, rhs_proj.reshape(ne, k - 1, ncomp))
            Pi_res = np.einsum('iac,aq->iqc', coef, psi)
            res -= Pi_res
        eta_vol_sq = np.einsum(
            'iqc,i,q->i',
            res ** 2,
            dx ** 3,
            integ.w
        )
        return SimpleNamespace(eta_global=np.sqrt(np.sum(eta_vol_sq))/np.pi/k, eta=eta_vol_sq)
    # -------------------------------------------------------------
    def compute_residual(self, mesh, uh):
        nx, k, ncomp, app = mesh.shape[0], self.korder, self.app.ncomp, self.app
        integ = self.integration['rhs']

        xm, dx = mesh[:-1], mesh[1:] - mesh[:-1]
        xq = xm[:, None] + dx[:, None] * integ.xi[None, :]

        uloc = self.local_coeffs(mesh, uh)

        u_vals = np.einsum('ijc,jq->iqc', uloc, integ.phi)
        du_ref = np.einsum('ijc,jq->iqc', uloc, integ.dphi)
        du_dx = du_ref / dx[:, None, None]

        f_all = app.f(xq, u_vals)
        if ncomp == 1 and f_all.ndim == 2:
            f_all = f_all[:, :, None]
        rhs_f = np.einsum(
            'i,q,iqc,jq->ijc',
            dx, integ.w, f_all, integ.phi
        )

        alpha = self.normalize_coefficient(app.diffusion_coef(xq), xq)
        Adu = self.apply_coefficient(alpha, du_dx)

        rhs_Au = np.einsum(
            'q,iqc,jq->ijc',
            integ.w, Adu, integ.dphi
        )
        # res_loc = rhs_f - rhs_Au
        # beta = self.normalize_coefficient(app.convection_coef(xq), xq)
        # Bdu = self.apply_coefficient(beta, du_dx)
        #
        # rhs_Bu = np.einsum(
        #     "i,q,iqc,jq->ijc",
        #     dx, integ.w, Bdu, integ.phi,
        #     optimize=True,
        # )
        #
        # res_loc = rhs_f - rhs_Au - rhs_Bu

        res_loc = rhs_f - rhs_Au

        res = self.make_fe_vector(mesh)
        res.p1[:-1, :] += res_loc[:, 0, :]
        res.p1[1:, :] += res_loc[:, k, :]
        if k > 1:
            res.bubbles[:, :, :] = res_loc[:, 1:k, :]
        res = self.dirichlet_zero(mesh, res)
        return res
    # -------------------------------------------------------------
    def rhs(self, mesh, uh):
        nx, k, ncomp, app = mesh.shape[0], self.korder, self.app.ncomp, self.app
        integ = self.integration['rhs']
        xm, dx = mesh[:-1],  (mesh[1:] - mesh[:-1])
        x_rhs = xm[:, None] + dx[:, None] * integ.xi[None, :]  # shape (nx-1, int_n)
        uloc = self.local_coeffs(mesh, uh)
        u_vals = np.einsum('ijc,jq->iqc', uloc, integ.phi)
        # --- Evaluate f(x,u) at all quadrature points ---
        # We collect all values and stack to (nx-1, int_n, ncomp)
        f_all = app.f(x_rhs, u_vals)
        if ncomp == 1 and f_all.ndim == 2:
            f_all = f_all[:, :, None]
        rhs_loc = np.einsum(
            'i,q,iqc,jq->ijc',
            dx, integ.w, f_all, integ.phi
        )
        rhs = self.make_fe_vector(mesh)
        rhs.p1[:-1, :] += rhs_loc[:, 0, :]
        rhs.p1[1:, :] += rhs_loc[:, k, :]
        if k > 1:
            rhs.bubbles[:, :, :] = rhs_loc[:, 1:k, :]
        # rhs.rhs_loc = rhs_loc
        return rhs
    def assemble_p1_global_fast(self, mesh, App):
        ncomp = self.app.ncomp
        nx = mesh.shape[0]
        ne = nx - 1

        # App shape: (ne, 2*ncomp, 2*ncomp)

        elem_nodes = np.column_stack([
            np.arange(ne),
            np.arange(1, ne + 1),
        ])

        elem_dofs = (
                elem_nodes[:, :, None] * ncomp
                + np.arange(ncomp)[None, None, :]
        ).reshape(ne, 2 * ncomp)

        rows = np.repeat(elem_dofs[:, :, None], 2 * ncomp, axis=2)
        cols = np.repeat(elem_dofs[:, None, :], 2 * ncomp, axis=1)

        A = sparse.coo_matrix(
            (App.ravel(), (rows.ravel(), cols.ravel())),
            shape=(nx * ncomp, nx * ncomp))

        return A.tocsr()

    def matrix_p1_tangent_fast(self, mesh, uh=None):
        app = self.app
        ncomp = app.ncomp
        nx = mesh.shape[0]
        ne = nx - 1

        integ = self.integration["coeff"]
        xi, w = integ.xi, integ.w

        h = mesh[1:] - mesh[:-1]
        x0 = mesh[:-1]
        xq = x0[:, None] + h[:, None] * xi[None, :]

        # P1 basis on reference element
        phi = np.vstack((1.0 - xi, xi))  # (2,nq)
        dphi = np.array([[-1.0] * len(xi),
                         [1.0] * len(xi)])  # (2,nq)

        ndloc = 2 * ncomp
        App = np.zeros((ne, ndloc, ndloc))

        # ------------------------------------------------------------
        # diffusion: int alpha u' v'
        # ------------------------------------------------------------
        Acoef = self.normalize_coefficient(app.diffusion_coef(xq), xq)

        if Acoef.ndim == xq.ndim:
            # scalar alpha
            A_s = np.einsum(
                "eq,aq,bq,q,e->eab",
                Acoef, dphi, dphi, w, 1.0 / h,
                optimize=True,
            )
            App += np.einsum(
                "eab,cd->eacbd",
                A_s, np.eye(ncomp),
                optimize=True,
            ).reshape(ne, ndloc, ndloc)
        else:
            # matrix alpha
            if Acoef.ndim == 2:
                Acoef = np.broadcast_to(Acoef, xq.shape + Acoef.shape)

            A = np.einsum(
                "eqcd,aq,bq,q,e->eacbd",
                Acoef, dphi, dphi, w, 1.0 / h,
                optimize=True,
            )
            App += A.reshape(ne, ndloc, ndloc)

        # ------------------------------------------------------------
        # convection: int beta u' v
        # no h factor: dx cancels derivative 1/h
        # ------------------------------------------------------------
        Bcoef = self.normalize_coefficient(app.convection_coef(xq), xq)

        if np.any(Bcoef != 0):
            if Bcoef.ndim == xq.ndim:
                B_s = np.einsum(
                    "eq,aq,bq,q->eab",
                    Bcoef, phi, dphi, w,
                    optimize=True,
                )
                App += np.einsum(
                    "eab,cd->eacbd",
                    B_s, np.eye(ncomp),
                    optimize=True,
                ).reshape(ne, ndloc, ndloc)
            else:
                if Bcoef.ndim == 2:
                    Bcoef = np.broadcast_to(Bcoef, xq.shape + Bcoef.shape)

                B = np.einsum(
                    "eqcd,aq,bq,q->eacbd",
                    Bcoef, phi, dphi, w,
                    optimize=True,
                )
                App += B.reshape(ne, ndloc, ndloc)

        # ------------------------------------------------------------
        # tangent reaction: - int df_du(x,u) p v
        # ------------------------------------------------------------
        if uh is not None:
            uL = uh.p1[:-1, :]
            uR = uh.p1[1:, :]
            u_vals = (
                    phi[0][None, :, None] * uL[:, None, :]
                    + phi[1][None, :, None] * uR[:, None, :]
            )

            J = app.df_du(xq, u_vals)
            J = self.normalize_jacobian(J, xq, u_vals)

            C = np.einsum(
                "eqcd,aq,bq,q,e->eacbd",
                J, phi, phi, w, h,
                optimize=True,
            )
            App -= C.reshape(ne, ndloc, ndloc)

        # left boundary row: element 0, local left node
        App[0, 0:ncomp, :] = 0.0
        for c in range(ncomp):
            App[0, c, c] = 1.0

        # right boundary row: last element, local right node
        r0 = ncomp
        App[-1, r0:r0 + ncomp, :] = 0.0
        for c in range(ncomp):
            App[-1, r0 + c, r0 + c] = 1.0
        Aglob = self.assemble_p1_global_fast(mesh, App)
        return SimpleNamespace(local=None, global_p1=Aglob)
    def matrix(self, mesh, uh=None):
        # if self.korder == 1:
        #     return self.matrix_p1_tangent_fast(mesh, uh)
        app = self.app
        k = self.korder
        ncomp = app.ncomp
        ne = mesh.shape[0] - 1

        integ = self.integration["coeff"]

        x0 = mesh[:-1]
        dx = mesh[1:] - mesh[:-1]
        xq = x0[:, None] + dx[:, None] * integ.xi[None, :]

        # ------------------------------------------------------------
        # Diffusion part
        # ------------------------------------------------------------
        Acoef = np.asarray(app.diffusion_coef(xq))

        if self.is_scalar_coefficient(Acoef, xq):
            # scalar diffusion coefficient: A(x) I_component
            A_diff_s = self.make_diffusion_local_scalar(mesh)  # (ne,k+1,k+1)

            A_diff = np.einsum(
                "eij,cd->eicjd",
                A_diff_s,
                np.eye(ncomp),
                optimize=True,
            )
            A_diff = A_diff.reshape(ne, (k + 1) * ncomp, (k + 1) * ncomp)

        else:
            # true coupled diffusion coefficient
            A_diff = self.make_diffusion_local_matrix(mesh)
            # expected shape: (ne,(k+1)*ncomp,(k+1)*ncomp)

        Aloc = A_diff.copy()

        # ------------------------------------------------------------
        # Convection part
        # term: int beta u' v
        # ------------------------------------------------------------
        # Bcoef = self.normalize_coefficient(app.convection_coef(xq), xq)
        # if np.any(Bcoef != 0):
        #     if Bcoef.ndim == xq.ndim:
        #         # scalar convection coefficient
        #         B_s = np.einsum(
        #             "eq,iq,jq,q,e->eij",
        #             Bcoef,
        #             integ.phi,  # test
        #             integ.dphi,  # trial derivative
        #             integ.w,
        #             np.ones_like(dx),
        #             optimize=True,
        #         )
        #
        #         B = np.einsum(
        #             "eij,cd->eicjd",
        #             B_s,
        #             np.eye(ncomp),
        #             optimize=True,
        #         )
        #         B = B.reshape(ne, (k + 1) * ncomp, (k + 1) * ncomp)
        #
        #     else:
        #         if Bcoef.ndim == 2:
        #             Bcoef = np.broadcast_to(Bcoef, xq.shape + Bcoef.shape)
        #
        #         B = np.einsum(
        #             "eqcd,iq,jq,q->eicjd",
        #             Bcoef,
        #             integ.phi,
        #             integ.dphi,
        #             integ.w,
        #             optimize=True,
        #         )
        #         B = B.reshape(ne, (k + 1) * ncomp, (k + 1) * ncomp)
        #
        #     Aloc += B

        # tangent contribution: - df_du(u)
        if uh is not None:
            uloc = self.local_coeffs(mesh, uh)
            u_vals = np.einsum("eic,iq->eqc", uloc, integ.phi)

            J = app.df_du(xq, u_vals)
            J = self.normalize_jacobian(J, xq, u_vals)

            C = np.einsum(
                "eqab,iq,jq,q,e->eiajb",
                J, integ.phi, integ.phi, integ.w, dx,
                optimize=True,
            )
            C = C.reshape(ne, (k + 1) * ncomp, (k + 1) * ncomp)

            Aloc -= C
        # ------------------------------------------------------------
        # Split local block into vertex/bubble blocks if k > 1
        # local ordering: node-major, i.e.
        #   (node 0 comp 0..ncomp-1),
        #   (node 1 comp 0..ncomp-1), ...
        # ------------------------------------------------------------
        Alocal = None if k == 1 else self.make_local_fe_matrix(mesh)

        vertex_ids = np.r_[
            np.arange(0, ncomp),
            np.arange(k * ncomp, (k + 1) * ncomp),
        ]
        bubble_ids = np.arange(ncomp, k * ncomp)

        App = Aloc[:, vertex_ids[:, None], vertex_ids]

        Aglob = self.assemble_p1_global_fast(mesh, App)

        if k > 1:
            Alocal.Apb[:, :, :] = Aloc[:, vertex_ids[:, None], bubble_ids]
            Alocal.Abp[:, :, :] = Aloc[:, bubble_ids[:, None], vertex_ids]
            Alocal.Abb[:, :, :] = Aloc[:, bubble_ids[:, None], bubble_ids]

        return SimpleNamespace(local=Alocal, global_p1=Aglob.tocsr())
    # -------------------------------------------------------------
    def dirichlet(self, mesh, uh):
        uh.p1[0, :] = self.app.uL
        uh.p1[-1, :] = self.app.uR
        return uh
    def dirichlet_zero(self, mesh, fh):
        fh.p1[0, :] = 0.0
        fh.p1[-1, :] = 0.0
        return fh
    def dirichlet_matrix(self, mesh, Ah):
        # if self.korder == 1: return Ah
        nx, k, ncomp = mesh.shape[0], self.korder, self.app.ncomp
        Aglobal, Alocal = Ah.global_p1, Ah.local
        bc = []
        for c in range(ncomp):
            bc.append(c)
            bc.append((nx - 1) * ncomp + c)
        if Alocal is not None:
            # Prevent condensation from modifying boundary P1 equations
            for c in range(ncomp):
                Alocal.Apb[0, c, :] = 0.0
                Alocal.Apb[-1, ncomp + c, :] = 0.0
        # Dirichlet P1
        # A = Aglobal.tolil()
        # for i in bc:
        #     A[i, :] = 0.0
        #     A[i, i] = 1.0
        # return SimpleNamespace(local=Alocal, global_p1=A.tocsc())
        A = Aglobal
        left = np.arange(ncomp)
        right = np.arange((nx - 1) * ncomp, nx * ncomp)
        bc = np.concatenate([left, right])
        starts = A.indptr[bc]
        ends = A.indptr[bc + 1]
        for s, e in zip(starts, ends):
            A.data[s:e] = 0.0
        for i, s, e in zip(bc, starts, ends):
            cols = A.indices[s:e]
            j = np.searchsorted(cols, i)
            A.data[s + j] = 1.0
        # for node in [0, nx - 1]:
        #     for c in range(ncomp):
        #         i = node * ncomp + c
        #         start, end = A.indptr[i], A.indptr[i + 1]
        #         A.data[start:end] = 0.0
        #         cols = A.indices[start:end]
        #         j = np.searchsorted(cols, i)
        #         assert j < len(cols) and cols[j] == i
        #         A.data[start + j] = 1.0
        return SimpleNamespace(local=Alocal, global_p1=A)

    # -------------------------------------------------------------
    def solve(self, mesh, uh=None):
        if uh is None:
            uh = self.make_fe_vector(mesh)
        fh = self.rhs(mesh, uh)
        # fh = self.dirichlet_zero(mesh, fh)
        fh = self.dirichlet(mesh, fh)
        return self.solve_linearized(mesh, uh, fh)
        Ah = self.matrix(mesh, uh)
        Ah = self.dirichlet_matrix(mesh, Ah)
        return self.linear_solver.solve(Ah, fh)

    # -------------------------------------------------------------
    def solve_linearized(self, mesh, uh, fh):
        with self.timer("matrix"):
            Ah = self.matrix(mesh, uh)
            Ah = self.dirichlet_matrix(mesh, Ah)
        with self.timer("linear_solver"):
            res = self.linear_solver.solve(Ah, fh)
        return res

#===========================================================================
if __name__ == '__main__':
    import elliptic_examples
    from Utility import mesh1d, plotting, logger
    import matplotlib.pyplot as plt
    import time

    def check_error(solver, niter=10, theta=0.5):
        ns, errL2, errH1, etas, times = [], [], [], [], []
        mesh = mesh1d.mesh(app.x0, app.x1, n=10)
        uh = solver.make_fe_vector(mesh)

        for iter in range(niter):
            t0 = time.time()
            ph = solver.solve(mesh)
            eL2, eH1 = solver.compute_error(mesh, ph)
            result = solver.compute_estimator_linearized(mesh, uh, ph)
            eta, eta_sq = result.deta_global, result.deta
            ns.append(len(mesh)), errL2.append(eL2), errH1.append(eH1), etas.append(eta), times.append(time.time()-t0)
            plot_dict = {app.name: {}}
            xplot, pplot = solver.eval_fe_on_grid(mesh, ph, nsub=50)
            # plot_dict['fineplot'] = {"x": xplot, "y": {"phh":pplot}}
            plot_dict[app.name]['x'] = mesh
            plot_dict[app.name]['y'] = {'ph': ph.p1}
            plot_dict[app.name]['kwargs'] = {'ph': {'marker': 'o'}}
            if hasattr(app, 'solution'):
                uex_nodes = app.solution(mesh)
                plot_dict[app.name]['y']['p'] = uex_nodes
                plot_dict[app.name]['y']['err'] = np.abs(uex_nodes - ph.p1)
            # if hasattr(app, 'solution'):
            #     plot_dict[app.name]['y']['p'] = app.solution(mesh)
            #     plot_dict[app.name]['y']['err'] = np.fabs(app.solution(mesh)[:, None] - ph.p1)
            plotting.add_mesh1d(plot_dict, mesh)
            plotting.plot_solutions(plot_dict)
            plt.show()
            mesh_new, maps = mesh1d.adapt_mesh(mesh, eta_sq, theta=theta)
            uh = solver.interpolate_to_new_mesh(mesh, uh, mesh_new, maps)
            ph = solver.interpolate_to_new_mesh(mesh, ph, mesh_new, maps)
            mesh = mesh_new

            # print(f"time: {time.time() - t0} error: {eH1} eta: {eta} rho = {eH1/eta}")
        ns = np.array(ns)
        errL2 = np.array(errL2)
        errH1 = np.array(errH1)
        etas = np.array(etas)
        times = np.array(times)

        def rates(ns, errs):
            return np.log(errs[:-1] / errs[1:]) / np.log(ns[1:] / ns[:-1])

        print("H1 rates:", rates(ns, errH1))
        print("L2 rates:", rates(ns, errL2))
        # pdict = {'x': ns, 'y': {'L2': errs_l2, 'disc': errs_disc, 'eta': etas}}
        pdict = {'x': ns, 'y': {'L2': errL2, 'H1': errH1, 'eta': etas}}
        plot_dict = {f"Errors {app.name} k={solver.korder}":  pdict}
        plotting.plot_error_curves(plot_dict)
        plt.show()


    # app = elliptic_examples.Poisson()
    # app = elliptic_examples.SmoothPoisson()
    # app = elliptic_examples.OscillatoryPoisson(omega=5, alpha=3.0)
    # app = elliptic_examples.DiscontinuousAlphaOscillator()
    app = elliptic_examples.InteriorLayerVariableAlpha()
    # app = elliptic_examples.LinearSystem3()
    solver = LinearElliptic(korder=1, app=app)
    check_error(solver, niter=21)
    print(solver.timer.summary(), '\n')
    print(solver.linear_solver.timer.summary())

