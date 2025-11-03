import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

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
        int_coeff = np.polynomial.legendre.leggauss(2*k+2)
        self.int_coeff_x, self.int_coeff_w = np.array(int_coeff[0]), np.array(int_coeff[1])
        int_rhs = np.polynomial.legendre.leggauss(k+2)
        self.int_rhs_x, self.int_rhs_w = np.array(int_rhs[0]), np.array(int_rhs[1])
        self.int_coeff_n, self.int_rhs_n = len(self.int_coeff_x), len(self.int_rhs_x)
        nbasis = k+1
        assert len(self.phi) == nbasis
        M = np.empty(shape=(nbasis,nbasis,self.int_coeff_n))
        Mold = np.empty(shape=(nbasis,self.int_coeff_n))
        B = np.empty(shape=(nbasis,self.int_rhs_n))
        Bold = np.empty(shape=(self.int_rhs_n))
        T = np.zeros(shape=(nbasis))
        T[0] = -1
        for i in range(1,nbasis): T[i] = 2.0/(1.0+2.0*i)
        # values in zero
        n_midpoint = (nbasis+1)//2
        self.midpoint = np.empty(shape=(n_midpoint))
        for k in range(n_midpoint):
            self.midpoint[k] = (-1)**k * math.comb(2*k, k) / 4**k
        for ii in range(self.int_coeff_n):
            w, x =self.int_coeff_w[ii], self.int_coeff_x[ii]
            for i in range(nbasis):
                Mold[i, ii] = w * (1.0 - self.phi[0](x)) * self.psi[i](x)
                for j in range(nbasis):
                    M[i, j, ii] = w * self.phi[i](x) * self.psi[j](x)
        for ii in range(self.int_rhs_n):
            w, x =self.int_rhs_w[ii], self.int_rhs_x[ii]
            Bold[ii] = w * (1.0 - self.phi[0](x))
            for i in range(nbasis):
                B[i, ii] = w * self.phi[i](x)
        self.M, self.T, self.B, self.Mold, self.Bold =  np.array(M), np.array(T), np.array(B), np.array(Mold), np.array(Bold)
        psi_val, dpsi_val = np.empty(shape=(self.int_coeff_n,nbasis)), np.empty(shape=(self.int_coeff_n,nbasis))
        for ii in range(self.int_coeff_n):
            w, x =self.int_coeff_w[ii], self.int_coeff_x[ii]
            for i in range(nbasis):
                psi_val[ii, i] = self.psi[i](x)
                dpsi_val[ii, i] = self.psi[i].deriv()(x)
        self.psival, self.dpsival = np.array(psi_val), np.array(dpsi_val)
        Mprolo = np.zeros(shape=(2,nbasis,nbasis))
        for ii in range(self.int_coeff_n):
            w, x =self.int_coeff_w[ii], self.int_coeff_x[ii]
            for i in range(nbasis):
                scale = 0.5*(2.0*i + 1.0)
                for j in range(nbasis):
                    Mprolo[0,i,j] += w*scale*self.psi[i](x)*self.psi[j](0.5*(x-1.0))
                    Mprolo[1,i,j] += w*scale*self.psi[i](x)*self.psi[j](0.5*(x+1.0))
        self.Mprolo = Mprolo
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
            # plt.plot(t, self.psi[i](t), '-', label=r"$\psi_{:1d}$".format(i))
        plt.legend()
        plt.grid()
        plt.subplot(122)
        for i in range(len(self.phi)):
            plt.plot(t, self.phi[i](t), '-', label=fr"$\phi_{{{i}}}$")
            # plt.plot(t, self.phi[i](t), '-', label=r"$\phi_{:1d}$".format(i))
        plt.legend()
        plt.grid()
        plt.show()
    def get_ab_values(self, app, t_coeff, t_rhs, nt, ncomp, nq_rhs, nq_coeff):
        # --- Precompute a_vals and b_vals if possible
        if hasattr(app, 'precompute_coeffs'):
            a_vals, b_vals = app.precompute_coeffs(t_coeff, t_rhs)
        else:
            b_vec = np.stack([app.b_coef(ti) for ti in t_rhs.flatten()])  # ( (nt-1)*nq_rhs, ncomp)
            b_vals = b_vec.reshape(nt - 1, nq_rhs, ncomp)  # (nt-1, nq_rhs, ncomp)
            a_vec = np.stack([app.a_coef(ti) for ti in t_coeff.flatten()])  # ((nt-1)*nq_coeff, ncomp, ncomp)
            a_vals = a_vec.reshape(nt - 1, nq_coeff, ncomp, ncomp)  # (nt-1, nq_coeff, ncomp, ncomp)
        return a_vals, b_vals
    def run(self, mesh, app):
        nt = mesh.shape[0]
        assert mesh.ndim == 1
        nbasis = len(self.phi)
        ncomp = 1 if np.ndim(app.u0) == 0 else len(app.u0)
        nq_coeff = self.int_coeff_n
        nq_rhs = self.int_rhs_n
        # Output arrays
        u_coef = np.empty((nt - 1, nbasis, ncomp), dtype=self.M.dtype)
        u_T = np.zeros(ncomp, dtype=self.M.dtype)
        bloc = np.zeros((nbasis, ncomp), dtype=self.M.dtype)
        Aloc = np.zeros((nbasis, nbasis, ncomp, ncomp), dtype=self.M.dtype)
        # --- Precompute quadrature points for all elements
        tm = 0.5 * (mesh[:-1] + mesh[1:])
        dt = 0.5 * (mesh[1:] - mesh[:-1])
        t_rhs = tm[:, None] + dt[:, None] * self.int_rhs_x[None, :]  # (nt-1, nq_rhs)
        t_coeff = tm[:, None] + dt[:, None] * self.int_coeff_x[None, :]  # (nt-1, nq_coeff)
        a_vals, b_vals = self.get_ab_values(app, t_coeff, t_rhs, nt, ncomp, nq_rhs, nq_coeff)
        # if hasattr(app, 'precompute_coeffs'):
        #     a_vals, b_vals = app.precompute_coeffs(t_coeff, t_rhs)
        # else:
        #     b_vec = np.stack([app.b_coef(ti) for ti in t_rhs.flatten()])  # ( (nt-1)*nq_rhs, ncomp)
        #     b_vals = b_vec.reshape(nt - 1, nq_rhs, ncomp)  # (nt-1, nq_rhs, ncomp)
        #     a_vec = np.stack([app.a_coef(ti) for ti in t_coeff.flatten()])  # ((nt-1)*nq_coeff, ncomp, ncomp)
        #     a_vals = a_vec.reshape(nt - 1, nq_coeff, ncomp, ncomp)  # (nt-1, nq_coeff, ncomp, ncomp)
        # --- Main element loop
        for it in range(nt - 1):
            # --- Right-hand side
            for ii in range(self.int_rhs_n):
                bloc[:, :] -= dt[it] * self.B[:, ii][:, None] * b_vals[it, ii]
            if it == 0:
                bloc[0] -= app.u0
            # --- Assemble local matrix
            Aloc.fill(0)
            Aloc[np.arange(nbasis), np.arange(nbasis)] = self.T[:, None, None] * np.eye(ncomp)
            Aloc += dt[it] * np.einsum('ijq,qab->ijab', self.M, a_vals[it])
            # --- Solve local system
            usol = np.linalg.solve(
                Aloc.swapaxes(1, 2).reshape(nbasis * ncomp, nbasis * ncomp),
                bloc.flat
            ).reshape((nbasis, ncomp))
            u_coef[it] = usol
            # --- Update u_T
            u_T = -self.T[0] * usol[0]
            u_T += dt[it] * np.tensordot(self.Bold, b_vals[it], axes=([0], [0]))
            for ii in range(self.int_coeff_n):
                u_T += dt[it] * np.sum(self.Mold[:, ii, None] * (a_vals[it, ii] @ usol.T).T, axis=0)
            # --- Prepare bloc for next element
            if it != nt - 1:
                bloc.fill(0)
                bloc[0] = -u_T
        return u_coef, u_T
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
        # nt = mesh.shape[0]
        nbasis, ncomp = u_h.shape[1], u_h.shape[2]
        # Quadrature for elements
        # nq = self.int_coeff_n
        x_q = np.array(self.int_coeff_x)  # shape (nq,)
        w_q = np.array(self.int_coeff_w)  # shape (nq,)
        # Midpoints and half-widths for all elements
        tm = 0.5 * (mesh[:-1] + mesh[1:])  # shape (nt-1,)
        dt = 0.5 * (mesh[1:] - mesh[:-1])  # shape (nt-1,)
        # Compute global quadrature points for all elements
        t_q = tm[:, None] + dt[:, None] * x_q[None, :]  # shape (nt-1, nq)
        # Evaluate solution at all quadrature points
        # sol_q = np.stack([solution(t_q) for t_q in t_q])  # shape (nt-1, nq, ncomp)
        # if sol_q.ndim == 1: sol_q = sol_q[:, None]
        sol_q = np.array([np.atleast_1d(solution(t)) for t in t_q])  # shape (nq, ncomp)
        if sol_q.ndim == 2:  sol_q = sol_q[:, :, None]
        # sol_a = np.stack([[np.atleast_1d(solution(t)) for t in tq] for tq in t_coeff])  # (nt-1, nq, ncomp)
        # Evaluate basis functions at quadrature points
        psi_eval = np.array([[psi(x) for psi in self.psi] for x in x_q])  # shape (nq, nbasis)
        u_approx = np.einsum('ebk,qb->eqk', u_h, psi_eval)
        # Compute element-wise error
        err = sol_q - u_approx  # shape (nt-1, nq, ncomp)
        # L2 error over elements
        err_l2 = np.sum(dt[:, None] * np.sum(w_q[None, :, None] * err ** 2, axis=1))
        # Store error per element
        err_cell = np.sum(dt[:, None] * np.sum(w_q[None, :, None] * err ** 2, axis=1), axis=1)
        # Discontinuous error using separate quadrature (int_disc)
        x_disc, w_disc = np.polynomial.legendre.leggauss(nbasis)  # shape (nbasis,)
        t_disc = tm[:, None] + dt[:, None] * x_disc[None, :]
        sol_disc = np.stack([solution(td) for td in t_disc])  # shape (nt-1, nq_disc, ncomp)
        psi_disc = np.array([[psi(x) for psi in self.psi] for x in x_disc])  # (nq_disc, nbasis)
        u_approx_disc = np.einsum('ebk,qb->eqk', u_h, psi_disc)
        if sol_disc.ndim == 2:  sol_disc = sol_q[:, :, None]
        err_disc_arr = sol_disc - u_approx_disc
        err_disc = np.sum(dt[:, None] * np.sum(w_disc[None, :, None] * err_disc_arr ** 2, axis=1))
        return np.sqrt(err_l2), np.sqrt(err_disc), err_cell
    def estimator(self, mesh, ucoeff, app):
        u_h, u_T = ucoeff
        nt = mesh.shape[0]
        ncomp = u_h.shape[-1]
        nq = len(self.int_coeff_x)
        # eta = np.zeros(shape=(nt-1), dtype=u_h.dtype)
        # h = 2*dt!!
        scale = 4/np.pi**2/(self.k+1.0)**2
        tm = 0.5 * (mesh[:-1] + mesh[1:])
        dt = 0.5 * (mesh[1:] - mesh[:-1])
        t_ii = tm[:, None] + dt[:, None] * self.int_coeff_x[None, :]
        u_ap = np.einsum('ejk,qj->eqk', u_h, self.psival)  # e=element, q=quad, k=dim
        du_ap = np.einsum('ejk,qj->eqk', u_h, self.dpsival)
        a_vals, b_vals = self.get_ab_values(app, t_ii, t_ii, nt, ncomp, nq, nq)
        res = np.einsum('eqab,eqb->eqa', a_vals, u_ap) + b_vals - du_ap / dt[:, None, None]

        # --- Integrate elementwise estimator
        w = self.int_coeff_w
        eta_cell = scale * np.sum(w[None, :] * dt[:, None] ** 3 * np.sum(res ** 2, axis=-1), axis=1)
        eta_global = np.sqrt(np.sum(eta_cell))
        return eta_global, eta_cell
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
    from Utility import mesh1d, plotting
    import ode_examples
    import matplotlib.pyplot as plt
    import time

    def check_error(app, k=0, n0=3, niter=6, plot=False, mesh_type="uniform"):
        cgp = ODE_Legendre(k=k)
        n = n0
        ns, errs_l2, errs_disc, etas = [], [], [], []
        for it in range(niter):
            n *=2
            ns.append(n)
            mesh = mesh1d.mesh(app.t_begin, app.t_end, n, type=mesh_type)
            t0 = time.time()
            u_h = cgp.run(mesh, app)
            t1 = time.time()
            el2, edisc, err = cgp.compute_error(mesh, u_h, app.solution)
            errs_l2.append(el2)
            errs_disc.append(edisc)
            t2 = time.time()
            eta, eta_cell = cgp.estimator(mesh, u_h, app)
            etas.append(eta)
            t3 = time.time()
            print(f"run:{t1-t0:8.2f}s err:{t2-t1:8.2f}s est:{t3-t2:8.2f}s {el2/eta:8.2f}")
            if plot:
                t_mp, u_mp = cgp.interpolate_midpoint(mesh, u_h)
                u_true = app.solution(t_mp)
                pd = {app.name:{}, "Error":{}, "Estimator":{}}
                pd[app.name]['x'] = t_mp
                pd[app.name]['y'] = {'app': u_mp, 'sol': u_true}
                pd[app.name]['kwargs'] = {'app':{'marker':'o'}}
                pd["Error"]['x'] = t_mp
                pd["Error"]['y'] = {'e': u_true-u_mp}
                pd["Estimator"]['x'] = t_mp
                pd["Estimator"]['y'] = {'eta': eta_cell, 'err': err}
                plotting.plot_solutions(pd)
                plt.show()
        ns = np.array(ns)
        errs_l2 = np.array(errs_l2)
        errs_disc = np.array(errs_disc)
        etas = np.array(etas)
        pdict = {'x': ns, 'y': {'L2': errs_l2, 'disc': errs_disc, 'eta': etas}}
        plot_dict = {"Errors":  pdict}
        plotting.plot_error_curves(plot_dict)
        plt.show()


    # app = ode_examples.PolynomialIntegration(degree=9, ncomp=2)
    # app = ode_examples.Exponential(lam=1.2)
    # app = ode_examples.ExponentialJordan()
    # app = ode_examples.TimeDependentRotation()
    app = ode_examples.RotScaleForce()
    check_error(app, k=2, niter=8, plot=True, mesh_type='random')

    # cgp = ODE_Legendre(k=3)
    # cgp.plot_basis()
