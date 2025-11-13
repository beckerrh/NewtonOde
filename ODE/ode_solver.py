# import jax
# jax.config.update("jax_enable_x64", True)
import numpy as np
from types import SimpleNamespace

#==================================================================
class ODE_Legendre():
#==================================================================
    def evaluate_on_integration_points(self, mesh, ucoeff):
        nt, n_integ = mesh.shape[0]-1, self.int_coeff_n
        nbasis, ncomp = ucoeff.shape[1], ucoeff.shape[2]
        meshp = np.empty(shape=(nt, n_integ))
        up = np.zeros(shape=(nt, n_integ, ncomp))
        for it in range(nt):
            tm, dt = 0.5 * (mesh[it + 1] + mesh[it]), 0.5 * (mesh[it + 1] - mesh[it])
            for ii in range(n_integ):
                w, x = self.int_coeff_w[ii], self.int_coeff_x[ii]
                meshp[it,ii] = tm + dt * x
                for i in range(nbasis):
                    up[it,ii] += ucoeff[it, i] * self.psi[i](x)
        return meshp, up
    def reference_integration(self, degree):
        x, w = np.polynomial.legendre.leggauss(degree)
        psivals = np.empty(shape=(len(x), self.k+1))
        for i in range(self.k+1):
            psivals[:, i] = self.psi[i](x)
        return SimpleNamespace(x=x, w=w, n=len(x), psi=psivals)
    def construct_basis_and_matrices(self, k:int):
        # we need order 2k+2 for coefficient and order k+2 for rhs
        import math
        self.k = k
        self.psi, self.dpsi, self.phi = [], [], []
        for i in range(k+1):
            # scale = np.sqrt((2*i+1)/2)
            # p = scale*np.polynomial.legendre.Legendre.basis(deg=i, domain=[-1, 1])
            p = np.polynomial.legendre.Legendre.basis(deg=i, domain=[-1, 1])
            self.psi.append(p)
            self.dpsi.append(p.deriv())
        q = 0.5*(np.polynomial.legendre.Legendre.basis(deg=0, domain=[-1, 1]) - np.polynomial.legendre.Legendre.basis(deg=1, domain=[-1, 1]))
        self.phi.append(q)
        for i in range(1,k+1):
            q = self.psi[i].integ()
            self.phi.append(q-q(-1))
            # assert self.phi[-1].deriv() == self.psi[-1]
        self.integration = {'coeff': self.reference_integration(2 * k + 2),
                            'rhs': self.reference_integration(k + 2),
                            'est': self.reference_integration(3 * k + 3)}
        nbasis = k+1
        assert len(self.phi) == nbasis

        integ_coef = self.integration['coeff']
        integ_rhs = self.integration['rhs']

        M = np.empty(shape=(nbasis, nbasis, integ_coef.n))
        Mold = np.empty(shape=(nbasis, integ_coef.n))
        B = np.empty(shape=(nbasis, integ_rhs.n))
        Bold = np.empty(shape=(integ_rhs.n))
        T = np.zeros(shape=(nbasis))
        T[0] = -1
        for i in range(1,nbasis): T[i] = 2.0/(1.0+2.0*i)
        # values in zero
        n_midpoint = (nbasis+1)//2
        self.midpoint = np.empty(shape=(n_midpoint))
        for k in range(n_midpoint):
            self.midpoint[k] = (-1)**k * math.comb(2*k, k) / 4**k
        for ii in range(integ_coef.n):
            w, x = integ_coef.w[ii], integ_coef.x[ii]
            for i in range(nbasis):
                Mold[i, ii] = w * (1.0 - self.phi[0](x)) * self.psi[i](x)
                for j in range(nbasis):
                    M[i, j, ii] = w * self.phi[i](x) * self.psi[j](x)
        for ii in range(integ_rhs.n):
            w, x = integ_rhs.w[ii], integ_rhs.x[ii]
            Bold[ii] = w * (1.0 - self.phi[0](x))
            for i in range(nbasis):
                B[i, ii] = w * self.phi[i](x)
        # print(f"{Mold=}\n {M=}")
        self.M, self.T, self.B, self.Mold, self.Bold =  M, T, B, Mold, Bold
        Mprolo = np.zeros(shape=(2,nbasis,nbasis))
        for ii in range(integ_coef.n):
            w, x = integ_coef.w[ii], integ_coef.x[ii]
            for i in range(nbasis):
                scale = 0.5*(2.0*i + 1.0)
                for j in range(nbasis):
                    Mprolo[0,i,j] += w*scale*self.psi[i](x)*self.psi[j](0.5*(x-1.0))
                    Mprolo[1,i,j] += w*scale*self.psi[i](x)*self.psi[j](0.5*(x+1.0))
        self.Mprolo = Mprolo
        j = np.arange(self.k)
        self.GdiagInv = (2 * j + 1.0)/2.0  # shape (nbp,)

    def __init__(self, k=0):
        self.name = self.__class__.__name__ + f"_{k}"
        self.construct_basis_and_matrices(k)
    def plot_basis(self):
        import matplotlib.pyplot as plt
        t = np.linspace(-1, 1)
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        for i in range(len(self.phi)):
            plt.plot(t, self.psi[i](t), '-', label=fr"$\psi_{{{i}}}$")
        plt.legend()
        plt.grid()
        plt.subplot(122)
        for i in range(len(self.phi)):
            plt.plot(t, self.phi[i](t), '-', label=fr"$\phi_{{{i}}}$")
        plt.legend()
        plt.grid()
        plt.show()
    def x_norm(self, mesh, xall):
        x, xT = xall
        dt = 0.5 * (mesh[1:] - mesh[:-1])
        integ = self.integration['coeff']
        x_vals = np.einsum('tjc,qj->tqc', x, integ.psi)
        return np.einsum('t,q,tqc->', dt, integ.w, x_vals**2)

    def compute_residual(self, mesh, app, xall):
        nt = mesh.shape[0]
        assert mesh.ndim == 1
        nbasis = len(self.phi)
        ncomp = 1 if np.ndim(app.x0) == 0 else len(app.x0)
        rhs1 = np.zeros(shape=(nt, ncomp), dtype=self.M.dtype)
        rhs2 = np.zeros(shape=(nt - 1, nbasis - 1, ncomp), dtype=self.M.dtype)
        tm, dt = 0.5 * (mesh[:-1] + mesh[1:]), 0.5 * (mesh[1:] - mesh[:-1])

        integ = self.integration['rhs']
        t_rhs = tm[:, None] + dt[:, None] * integ.x[None, :]  # shape (nt-1, int_n)
        nint = integ.n

        x, xT = xall
        # x_vals shape (nt-1, int_n, ncomp)
        x_vals = np.einsum('ijc,qj->iqc', x, integ.psi)

        # --- Evaluate f(t,u) at all quadrature points ---
        # We collect all values and stack to (nt-1, int_n, ncomp)
        f_all = np.zeros_like(x_vals)
        for it in range(nt - 1):
            for ii in range(nint):
                f_all[it, ii] = app.f(t_rhs[it, ii], x_vals[it, ii])

        # --- Quadrature (vectorized) ---
        rhs1[0] = -app.x0
        rhs1[:-1] -= np.einsum('t,i,tic->tc', dt, self.B[0], f_all)
        rhs1[1:] -= np.einsum('t,i,tic->tc', dt, self.Bold, f_all)
        rhs2 -= np.einsum('t,bi,tic->tbc', dt, self.B[1:], f_all)

        # --- Derivative contributions ---
        # print(f"{nt=} {self.T[0]=} {x[:-1, 0].shape=} {rhs1[:-1].shape=}")
        rhs1[:-1] -= self.T[0] * x[:, 0]
        rhs1[1:] += self.T[0] * x[:, 0]
        rhs2 -= np.einsum('b,tbq->tbq', self.T[1:], x[:, 1:])
        # print(f"rhs1[-1] = {rhs1[-1]} {xT=}")
        rhs1[-1] += xT
        return rhs1, rhs2

    def get_a_linearized(self, mesh, app, xall):
        x, xT = xall
        nt = mesh.shape[0]
        ncomp = 1 if np.ndim(app.x0) == 0 else len(app.x0)
        integ = self.integration['coeff']
        tm, dt = 0.5 * (mesh[:-1] + mesh[1:]), 0.5 * (mesh[1:] - mesh[:-1])
        t_coeff = tm[:, None] + dt[:, None] * integ.x[None, :]  # (nt-1, self.int_coeff_n)
        x_vals = np.einsum('ijc,qj->iqc', x, integ.psi)
        A_vals = np.stack([app.df(t, u) for t, u in zip(t_coeff.ravel(), x_vals.reshape(-1, ncomp))])
        A_vals = A_vals.reshape(nt - 1, integ.n, ncomp, ncomp)
        a0_vals = np.einsum('t,iq,tqnm->tinm', dt, self.Mold, A_vals)
        a_vals = np.einsum('t,ijq,tqnm->tijnm', dt, self.M, A_vals)
        return a_vals, a0_vals
    def solve_linearized(self, mesh, app, x, res):
        rhs1, rhs2 = res
        nt = mesh.shape[0]
        assert mesh.ndim == 1
        nbasis = len(self.phi)
        ncomp = 1 if np.ndim(app.x0) == 0 else len(app.x0)
        x_coef = np.empty((nt - 1, nbasis, ncomp), dtype=self.M.dtype)
        bloc = np.zeros((nbasis, ncomp), dtype=self.M.dtype)
        Aloc_template = np.zeros((nbasis, nbasis, ncomp, ncomp), dtype=self.M.dtype)
        for i in range(nbasis):
            Aloc_template[i, i, :, :] = np.eye(ncomp) * self.T[i]
        a_vals, a0_vals = self.get_a_linearized(mesh, app, x)
        for it in range(nt - 1):
            bloc[0,:] += rhs1[it]
            bloc[1:,:] += rhs2[it]
            Aloc = Aloc_template + a_vals[it]
            Aflat = Aloc.swapaxes(1, 2).reshape(nbasis * ncomp, nbasis * ncomp)
            # --- Solve local system
            xsol = np.linalg.solve(Aflat, bloc.ravel()).reshape((nbasis, ncomp))
            # print(f"{it=}: {bloc.squeeze()=} {Aloc.squeeze()=} {usol.squeeze()=}")
            x_coef[it] = xsol
            bloc.fill(0)
            bloc[0] = self.T[0] * xsol[0] - np.einsum('icd,id->c', a0_vals[it], xsol)
        x_T = - bloc[0] - rhs1[-1]
        return x_coef, x_T
    def solve_linear(self, mesh, app):
        ncomp = 1 if np.ndim(app.x0) == 0 else len(app.x0)
        x0 = np.zeros(shape=(len(mesh) - 1, self.k + 1, ncomp), dtype=mesh.dtype)
        x = (x0, np.zeros(x0.shape[2]))
        res = self.compute_residual(mesh, app, x)
        # print(f"res = {res}")
        du_h = self.solve_linearized(mesh, app, x, res)
        # print(f"du_h = {du_h}")
        xnew = (x[0] + du_h[0], x[1] + du_h[1])
        # print(f"xnew = {xnew}")
        res = self.compute_residual(mesh, app, xnew)
        print(f"res = {np.linalg.norm(res[0])+np.linalg.norm(res[1])}")
        return xnew

    def _get_a_linearized(self, it, tm, dt, mesh, app, xall):
        x, xT = xall
        nt = mesh.shape[0]
        ncomp = 1 if np.ndim(app.x0) == 0 else len(app.x0)
        integ = self.integration['coeff']
        t_coeff = tm + dt * integ.x[:]
        x_vals = np.einsum('jc,qj->qc', x[it], integ.psi)
        A_vals = np.stack([app.df(t, u) for t, u in zip(t_coeff.ravel(), x_vals.reshape(-1, ncomp))])
        A_vals = A_vals.reshape(integ.n, ncomp, ncomp)
        a0_vals = dt *np.einsum('iq,qnm->inm', self.Mold, A_vals)
        a_vals = dt *np.einsum('ijq,qnm->ijnm', self.M, A_vals)
        return a_vals, a0_vals
    def _compute_residual(self, it, tm, dt, mesh, app, xall):
        nt = mesh.shape[0]
        assert mesh.ndim == 1
        nbasis = len(self.phi)
        ncomp = 1 if np.ndim(app.x0) == 0 else len(app.x0)
        rhs1 = np.zeros(shape=(ncomp), dtype=self.M.dtype)
        rhs1old = np.zeros(shape=(ncomp), dtype=self.M.dtype)
        rhs2 = np.zeros(shape=(nbasis - 1, ncomp), dtype=self.M.dtype)
        # tm, dt = 0.5 * (mesh[:-1] + mesh[1:]), 0.5 * (mesh[1:] - mesh[:-1])

        integ = self.integration['rhs']
        t_rhs = tm + dt * integ.x
        nint = integ.n

        x, xT = xall
        # x_vals shape (nt-1, int_n, ncomp)
        x_vals = np.einsum('jc,qj->qc', x[it], integ.psi)

        # --- Evaluate f(t,u) at all quadrature points ---
        # We collect all values and stack to (nt-1, int_n, ncomp)
        f_all = np.zeros_like(x_vals)
        for ii in range(nint):
            f_all[ii] = app.f(t_rhs[ii], x_vals[ii])

        # --- Quadrature (vectorized) ---
        if it==0: rhs1 = -app.x0
        rhs1 -= dt*np.einsum('i,ic->c', self.B[0], f_all)
        rhs1old = -dt*np.einsum('i,ic->c',  self.Bold, f_all)
        rhs2 = -dt*np.einsum('bi,ic->bc', self.B[1:], f_all)

        # --- Derivative contributions ---
        # print(f"{nt=} {self.T[0]=} {x[:-1, 0].shape=} {rhs1[:-1].shape=}")
        rhs1 -= self.T[0] * x[it][0]
        rhs1old += self.T[0] * x[it][0]
        rhs2 -= np.einsum('b,bq->bq', self.T[1:], x[it,1:])
        # print(f"rhs1[-1] = {rhs1[-1]} {xT=}")
        rhs1[-1] += xT
        return rhs1, rhs2
    def solve_semi_implicit(self, mesh, app, x):
        rhs1_all, rhs2_all = self.compute_residual(mesh, app, x)
        nt = mesh.shape[0]
        assert mesh.ndim == 1
        nbasis = len(self.phi)
        ncomp = 1 if np.ndim(app.x0) == 0 else len(app.x0)
        x_coef = np.empty((nt - 1, nbasis, ncomp), dtype=self.M.dtype)
        x_T = np.zeros(ncomp, dtype=self.M.dtype)
        bloc = np.zeros((nbasis, ncomp), dtype=self.M.dtype)
        Aloc_template = np.zeros((nbasis, nbasis, ncomp, ncomp), dtype=self.M.dtype)
        for i in range(nbasis):
            Aloc_template[i, i, :, :] = np.eye(ncomp) * self.T[i]
        # a_vals, a0_vals = self.get_a_linearized(mesh, app, x)
        tm, dt = 0.5 * (mesh[:-1] + mesh[1:]), 0.5 * (mesh[1:] - mesh[:-1])
        for it in range(nt - 1):
            rhs1, rhs2 = self._compute_residual(it, tm[it], dt[it], mesh, app, x)
            bloc[0,:] += rhs1
            bloc[1:,:] += rhs2
            a_vals, a0_vals = self._get_a_linearized(it, tm[it], dt[it], mesh, app, x)
            Aloc = Aloc_template + a_vals
            Aflat = Aloc.swapaxes(1, 2).reshape(nbasis * ncomp, nbasis * ncomp)
            # --- Solve local system
            xsol = np.linalg.solve(Aflat, bloc.ravel()).reshape((nbasis, ncomp))
            # print(f"{it=}: {bloc.squeeze()=} {Aloc.squeeze()=} {usol.squeeze()=}")
            x_coef[it] = xsol
            x_T = np.einsum('icd,id->c', a0_vals, xsol) - self.T[0] * xsol[0]
            if it < nt - 2:
                bloc.fill(0)
                bloc[0] = -x_T
            else:
                x_T -= rhs1_all[it + 1]
        return x_coef, x_T
    def interpolate_midpoint(self, mesh, ucoeff):
        u_h, u_T = ucoeff
        # print(f"{u_h.shape=} {u_T.shape=}")
        nt = mesh.shape[0]
        nbasis, ncomp = u_h.shape[1], u_h.shape[2]
        # nt, ncomp, nbasis = mesh.shape[0], u_h.shape[-1], u_h.shape[1]
        assert u_h.shape == (nt-1, nbasis, ncomp)
        u_mp = np.zeros(shape=(nt-1, ncomp), dtype=u_h.dtype)
        for it in range(0,nt-1):
            for ii in range(len(self.midpoint)):
                u_mp[it] += self.midpoint[ii] * u_h[it, 2*ii]
        return 0.5*(mesh[1:]+mesh[:-1]), u_mp
    def compute_error(self, mesh, ucoeff, solution):
        u_h, u_T = ucoeff
        nbasis, ncomp = u_h.shape[1], u_h.shape[2]
        integ = self.integration['coeff']
        # x_q = np.array(self.int_coeff_x)  # shape (nq,)
        # w_q = np.array(self.int_coeff_w)  # shape (nq,)
        # Midpoints and half-widths for all elements
        tm = 0.5 * (mesh[:-1] + mesh[1:])  # shape (nt-1,)
        dt = 0.5 * (mesh[1:] - mesh[:-1])  # shape (nt-1,)
        # Compute global quadrature points for all elements
        t_q = tm[:, None] + dt[:, None] * integ.x[None, :]  # shape (nt-1, nq)
        # Evaluate solution at all quadrature points
        sol_q = np.array([np.atleast_1d(solution(t)) for t in t_q])  # shape (nq, ncomp)
        if sol_q.ndim == 2:  sol_q = sol_q[:, :, None]
        # Evaluate basis functions at quadrature points
        psi_eval = np.array([[psi(x) for psi in self.psi] for x in integ.x])  # shape (nq, nbasis)
        u_approx = np.einsum('ebk,qb->eqk', u_h, psi_eval)
        # Compute element-wise error
        err = sol_q - u_approx  # shape (nt-1, nq, ncomp)
        # L2 error over elements
        err_l2 = np.sum(dt[:, None] * np.sum(integ.w[None, :, None] * err ** 2, axis=1))
        # Store error per element
        err_cell = np.sum(dt[:, None] * np.sum(integ.w[None, :, None] * err ** 2, axis=1), axis=1)
        return np.sqrt(err_l2), err_cell
    def _compute_cellwide_L2projection(self, res, integ, dt):
        psi_eval = integ.psi  # shape (nq, nbasis)
        psi_proj = psi_eval[:, :self.k]  # (nq, nbp)
        alpha = np.einsum('q,j,qj,tqc->tjc', integ.w, self.GdiagInv, psi_proj, res)
        # reconstruct projection at quadrature points: proj_{t,q,c} = psi_proj[q,j] * alpha_{t,j,c}
        proj = np.einsum('qj,tjc->tqc', psi_proj, alpha)  # (nt-1, nq, ncomp)
        # remainder and L2 per element (same as before)
        rtil = res - proj
        r2_sumc = np.sum(rtil ** 2, axis=2)  # (nt-1, nq)
        return np.einsum('t, q,tq->t', dt ** 3, integ.w, r2_sumc)
    def estimator(self, mesh, xall, pall, app):
        x, xT = xall
        p, pT = pall
        nt = mesh.shape[0]
        nbasis, ncomp = x.shape[1], x.shape[2]
        # h = 2*dt!!
        scale = 4.0/np.pi**2/(self.k+1.0)**2
        tm = 0.5 * (mesh[:-1] + mesh[1:])
        dt = 0.5 * (mesh[1:] - mesh[:-1])
        integ = self.integration['est']
        x_vals = np.einsum('ejk,qj->eqk', x, integ.psi)  # e=element, q=quad, k=dim
        p_vals = np.einsum('ejk,qj->eqk', p, integ.psi)  # e=element, q=quad, k=dim
        t_est = tm[:, None] + dt[:, None] * integ.x[None, :]  # (nt-1, int_n)
        res_eta = np.zeros(shape=(nt-1, integ.n, ncomp), dtype=self.M.dtype)
        res_zeta = np.zeros(shape=(nt-1, integ.n, ncomp), dtype=self.M.dtype)
        for it in range(nt - 1):
            for ii in range(integ.n):
                f_vals = app.f(t_est[it, ii], x_vals[it, ii])
                df_vals = app.df(t_est[it, ii], x_vals[it, ii])
                res_eta[it, ii] = f_vals
                res_zeta[it, ii] = f_vals + df_vals @ p_vals[it, ii]
        zeta_cell = scale * self._compute_cellwide_L2projection(res_zeta, integ, dt)
        eta_cell = scale * self._compute_cellwide_L2projection(res_eta, integ, dt)
        zeta_global = np.sqrt(np.sum(zeta_cell))
        eta_global = np.sqrt(np.sum(eta_cell))
        return zeta_global, zeta_cell, eta_global, eta_cell
    def interpolate(self, ucoeff, mesh_new, refinfo):
        u_h, u_T = ucoeff
        # print(f"{mesh=}\n{mesh_new=}")
        refined_map, non_refined_map = refinfo
        n_new = len(mesh_new) - 1
        u_h_new = np.empty((n_new, *u_h.shape[1:]), dtype=u_h.dtype)
        # --- Non-refined intervals: copy directly
        if len(non_refined_map):
            i_old = non_refined_map[:, 0]
            i_new = non_refined_map[:, 1]
            u_h_new[i_new] = u_h[i_old]
        # --- Refined intervals: project onto two subintervals
        if len(refined_map):
            i_old = refined_map[:, 0]
            i_newL = refined_map[:, 1]
            i_newR = refined_map[:, 2]
            M_left, M_right = self.Mprolo
            # use einsum for (n_refine, n_dof, ncomp)
            u_h_old = u_h[i_old]
            u_h_new[i_newL] = np.einsum('jk,ikr->ijr', M_left, u_h_old)
            u_h_new[i_newR] = np.einsum('jk,ikr->ijr', M_right, u_h_old)
        return u_h_new, u_T
    def plot_dg(self, mesh, ucoeff, kwargs={}):
        import matplotlib.pyplot as plt
        label = kwargs.pop('label', None)
        mp, up = self.evaluate_on_integration_points(mesh, ucoeff[0])
        if label:
            plt.plot(mp[0], up[0], **kwargs, label=label)
            for k in range(1,mp.shape[0]):
                plt.plot(mp[k], up[k], **kwargs)
        else:
            for k in range(mp.shape[0]):
                plt.plot(mp[k], up[k], **kwargs)

#------------------------------------------------------------------
if __name__ == "__main__":
    from Utility import mesh1d, plotting, printer
    import ode_examples
    import matplotlib.pyplot as plt
    import time

    def check_error(app, k=0, n0=10, niter=6, plot=False, mesh_type="uniform"):
        cgp = ODE_Legendre(k=k)
        n = n0
        ns, errs_l2, errs_disc, etas = [], [], [], []
        types = {'n': 'i', 'solve': 'f', 'err': 'f', 'est': 'f', 'eff': 'f'}
        pr = printer.Printer(types=types)
        pr.print_names()
        ncomp = 1 if np.ndim(app.x0) == 0 else len(app.x0)
        for it in range(niter):
            n *=2
            ns.append(n)
            mesh = mesh1d.mesh(app.t_begin, app.t_end, n, type=mesh_type)
            t0 = time.time()
            x0 = np.zeros(shape=(len(mesh) - 1, cgp.k+1, ncomp), dtype=mesh.dtype)
            x = (x0, np.zeros(x0.shape[2]))
            res = cgp.compute_residual(mesh, app, x)
            # print(f"{np.linalg.norm(res[0])=} {np.linalg.norm(res[1])=}")
            p = cgp.solve_linearized(mesh, app, x, res)
            p2 = cgp.solve_semi_implicit(mesh, app, x)
            assert np.allclose(p[0], p2[0]), "problem in 0"
            assert np.allclose(p[1], p2[1]), "problem in 1"
            # print(f"{np.linalg.norm(du_h[0])=}")
            # print(f"{u_h[0].squeeze()=}\n{du_h[0].squeeze()=}")
            x = (x[0] + p[0], x[1]+p[1])
            t1 = time.time()
            el2, err = cgp.compute_error(mesh, x, app.solution)
            errs_l2.append(el2)
            t2 = time.time()
            zeta, zeta_cell, eta, eta_cell = cgp.estimator(mesh, x, p, app)
            etas.append(eta)
            t3 = time.time()
            pr.values = {'n': n, 'solve': t1-t0, 'err': t2-t1, 'est': t3-t2, 'eff':el2/eta}
            pr.print()
            if plot:
                t_mp, u_mp = cgp.interpolate_midpoint(mesh, x)
                u_true = app.solution(t_mp)
                pd = {app.name:{}, "Estimator":{}}
                pd[app.name]['x'] = t_mp
                pd[app.name]['y'] = {'app': u_mp, 'sol': u_true}
                pd[app.name]['kwargs'] = {'app':{'marker':'o'}}
                # pd["Error"]['x'] = t_mp
                # pd["Error"]['y'] = {'e': u_true-u_mp}
                pd["Estimator"]['x'] = t_mp
                pd["Estimator"]['y'] = {'zeta': zeta_cell, 'eta': eta_cell, 'err': np.abs(err)}
                plotting.plot_solutions(pd)
                plt.show()
        ns = np.array(ns)
        print(f"{errs_l2=}")
        errs_l2 = np.array(errs_l2)
        etas = np.array(etas)
        # pdict = {'x': ns, 'y': {'L2': errs_l2, 'disc': errs_disc, 'eta': etas}}
        pdict = {'x': ns, 'y': {'L2': errs_l2, 'eta': etas}}
        plot_dict = {f"Errors {app.name}":  pdict}
        plotting.plot_error_curves(plot_dict)
        plt.show()


    # app = ode_examples.PolynomialIntegration(degree=9, ncomp=2)
    # app = ode_examples.Exponential(lam=1.2)
    # app = ode_examples.ExponentialJordan()
    app = ode_examples.TimeDependentShear()
    # app = ode_examples.TimeDependentRotation(t_end = 24)
    # app.check_linear()
    check_error(app, k=2, niter=12, n0=3, plot=False, mesh_type='random')

    # cgp = ODE_Legendre(k=3)
    # cgp.plot_basis()
