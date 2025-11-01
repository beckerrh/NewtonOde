import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial

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
        self.int_coeff_x, self.int_coeff_w = jnp.array(int_coeff[0]), jnp.array(int_coeff[1])
        int_rhs = np.polynomial.legendre.leggauss(k+2)
        self.int_rhs_x, self.int_rhs_w = jnp.array(int_rhs[0]), jnp.array(int_rhs[1])
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
        self.M, self.T, self.B, self.Mold, self.Bold =  jnp.array(M), jnp.array(T), jnp.array(B), jnp.array(Mold), jnp.array(Bold)
        psi_val, dpsi_val = np.empty(shape=(self.int_coeff_n,nbasis)), np.empty(shape=(self.int_coeff_n,nbasis))
        for ii in range(self.int_coeff_n):
            w, x =self.int_coeff_w[ii], self.int_coeff_x[ii]
            for i in range(nbasis):
                psi_val[ii, i] = self.psi[i](x)
                dpsi_val[ii, i] = self.psi[i].deriv()(x)
        self.psival, self.dpsival = jnp.array(psi_val), jnp.array(dpsi_val)
        Mprolo = np.zeros(shape=(2,nbasis,nbasis))
        for ii in range(self.int_coeff_n):
            w, x =self.int_coeff_w[ii], self.int_coeff_x[ii]
            for i in range(nbasis):
                scale = 0.5*(2.0*i + 1.0)
                for j in range(nbasis):
                    Mprolo[0,i,j] += w*scale*self.psi[i](x)*self.psi[j](0.5*(x-1.0))
                    Mprolo[1,i,j] += w*scale*self.psi[i](x)*self.psi[j](0.5*(x+1.0))
        print(f"{Mprolo[0]=}\n{Mprolo[1]=}")
        self.Mprolo = Mprolo

    def __init__(self, k=0):
        self.name = self.__class__.__name__ + f"_{k}"
        self.construct_basis_and_matrices(k)
    def plot_basis(self):
        import matplotlib.pyplot as plt
        t = jnp.linspace(-1, 1)
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
    def run(self, mesh, u0, a_coef, b_coef):
        nt = mesh.shape[0]
        assert mesh.ndim ==1
        nbasis = len(self.phi)
        ncomp = 1 if np.ndim(u0) == 0 else len(u0)
        print(f"{self.M.dtype=} {ncomp=} {b_coef(0).shape=}")
        assert a_coef(0).shape == (ncomp, ncomp)
        if ncomp>1: assert b_coef(0).shape[0] == ncomp
        u_coef = np.empty(shape=(nt - 1, nbasis, ncomp), dtype=self.M.dtype)
        u_T = np.zeros(shape=(ncomp), dtype=self.M.dtype)
        bloc = np.zeros(shape=(nbasis, ncomp), dtype=u_coef.dtype)
        Aloc = np.zeros((nbasis, nbasis, ncomp, ncomp), dtype=u_coef.dtype)
        for it in range(nt-1):
            tm, dt = 0.5*(mesh[it+1]+mesh[it]), 0.5*(mesh[it+1]-mesh[it])
            assert(dt>0)
            for ii in range(self.int_rhs_n):
                w, x = self.int_rhs_w[ii], self.int_rhs_x[ii]
                t_ii = tm + dt*x
                b_app = b_coef(t_ii)
                for i in range(nbasis):
                    bloc[i] -= dt*self.B[i,ii]*b_app
            if it==0:
                bloc[0] -= u0
            Aloc.fill(0)
            # print(f"{Aloc=}")
            for i in range(nbasis):
                Aloc[i, i] += self.T[i]*np.eye(ncomp)
            # print(f"{Aloc=}")
            for ii in range(self.int_coeff_n):
                w, x = self.int_coeff_w[ii], self.int_coeff_x[ii]
                t_ii = tm + dt*x
                a_app = a_coef(t_ii)
                # print(f"{dt=} {self.T=} {self.M=} {self.Mold=}")
                for i in range(nbasis):
                    for j in range(nbasis):
                        Aloc[i,j] += dt*self.M[i,j,ii]*a_app
            usol = np.linalg.solve(Aloc.swapaxes(1,2).reshape(nbasis*ncomp,nbasis*ncomp), bloc.flat).reshape((nbasis,ncomp))
            u_coef[it] = usol

            # Aloc_block = Aloc.reshape(nbasis * ncomp, nbasis * ncomp, order='F')
            # bloc_flat = bloc.reshape(nbasis * ncomp, order='F')
            # u_flat = np.linalg.solve(Aloc_block, bloc_flat)
            # u_coef[it] = u_flat.reshape(nbasis, ncomp, order='F')

            # print(f"{Aloc=} {bloc=} {usol=}")
            u_T = -self.T[0] * u_coef[it,0]
            for ii in range(self.int_rhs_n):
                w, x = self.int_rhs_w[ii], self.int_rhs_x[ii]
                t_ii = tm + dt * x
                b_app = b_coef(t_ii)
                u_T += dt*self.Bold[ii]*b_app
            for ii in range(self.int_coeff_n):
                w, x = self.int_coeff_w[ii], self.int_coeff_x[ii]
                t_ii = tm + dt*x
                a_app = a_coef(t_ii)
                for j in range(nbasis):
                    u_T += dt*self.Mold[j,ii]*(a_app@u_coef[it,j])
            if it != nt - 1:
                bloc.fill(0)
                # print(f"{bloc.shape=} {u_T.shape=} {u_coef[it,0].shape=} {b_app.shape=} {a_app.shape=}")
                bloc[0] = -u_T
        # print(f"{u_coef=}")
        # help = np.empty_like(u_coef)
        # help[:,:,0] = u_coef[:,:,1]
        # help[:,:,1] = u_coef[:,:,0]
        # return help, u_T
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
        nt = mesh.shape[0]
        nbasis, ncomp = u_h.shape[1], u_h.shape[2]
        err_cell = np.zeros(shape=(nt-1), dtype=u_h.dtype)
        err_l2, err_disc = 0.0, 0.0
        int_disc = np.polynomial.legendre.leggauss(nbasis)
        for it in range(nt-1):
            tm, dt = 0.5*(mesh[it+1]+mesh[it]), 0.5*(mesh[it+1]-mesh[it])
            for ii in range(self.int_coeff_n):
                w, x = self.int_coeff_w[ii], self.int_coeff_x[ii]
                t_ii = tm + dt * x
                sol = solution(t_ii)
                u_approx = np.zeros(shape=(ncomp))
                for i in range(nbasis):
                    u_approx += u_h[it,i]*self.psi[i](x)
                err = sol - u_approx
                err_l2 += w*dt*np.sum(err**2)
                err_cell[it] += w * dt * np.sum(err ** 2)
            for x,w in zip(int_disc[0], int_disc[1]):
                t_ii = tm + dt * x
                sol = solution(t_ii)
                u_approx = np.zeros(shape=(ncomp))
                for i in range(nbasis):
                    u_approx += u_h[it,i]*self.psi[i](x)
                err = sol - u_approx
                # print(f"{err=} {u_approx=} {sol=}")
                err_disc += w*dt*np.sum(err**2)
        return np.sqrt(err_l2), np.sqrt(err_disc), err_cell
    # @partial(jax.jit, static_argnums=(0, 3, 4))
    def estimator(self, mesh, ucoeff, a_coef, b_coef):
        u_h, u_T = ucoeff
        nt = mesh.shape[0]
        eta = np.zeros(shape=(nt-1), dtype=u_h.dtype)
        for it in range(nt-1):
            tm, dt = 0.5*(mesh[it+1]+mesh[it]), 0.5*(mesh[it+1]-mesh[it])
            for ii in range(self.int_coeff_n):
                w, x = self.int_coeff_w[ii], self.int_coeff_x[ii]
                t_ii = tm + dt * x
                # u_ap = u_h[it]*self.psival[ii]
                u_ap = np.einsum('jk,j->k', u_h[it], self.psival[ii])
                du_ap = np.einsum('jk,j->k', u_h[it], self.dpsival[ii])
                res = a_coef(t_ii)@u_ap + b_coef(t_ii) - du_ap/dt
                eta[it] += w*dt**3*jnp.sum(res**2)
                # eta.at[it].add(w * dt ** 3 * jnp.sum(res ** 2))
        # print(f"eta = {eta}")
        return jnp.sqrt(np.sum(eta))/np.pi/(self.k+1.0), eta

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
            u_h = cgp.run(mesh, app.u0, app.a_coef, app.b_coef)
            t1 = time.time()
            el2, edisc, err = cgp.compute_error(mesh, u_h, app.solution)
            errs_l2.append(el2)
            errs_disc.append(edisc)
            t2 = time.time()
            eta, eta_cell = cgp.estimator(mesh, u_h, app.a_coef, app.b_coef)
            etas.append(eta)
            t3 = time.time()
            print(f"{t1-t0:8.2f}s {t2-t1:8.2f}s {t3-t2:8.2f}s {el2/eta:8.2f}")
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


    # app = ode_examples.Logistic()
    app = ode_examples.PolynomialIntegration(degree=9, ncomp=2)
    # app = ode_examples.Exponential(lam=1.2)
    # app = ode_examples.ExponentialJordan()
    # app = ode_examples.TimeDependentRotation()
    # app = ode_examples.RotScaleForce()
    check_error(app, k=2, niter=6, plot=True, mesh_type='random')

    # cgp = ODE_Legendre(k=3)
    # cgp.plot_basis()
