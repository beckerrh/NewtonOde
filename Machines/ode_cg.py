import numpy as np
import sys

#==================================================================
class CgK():
#==================================================================
    """
    cgp on [-1,1]
    """
    def __init__(self, k=1, plotbasis=False, alpha=0):
        # alpha=0: CN alpha=1: Euler
        self.alpha = alpha
        self.error_type = "H1"
        self.name = self.__class__.__name__ + f"_{k}"
        assert k>=1
        self.k = k
        self.int_x, self.int_w = np.polynomial.legendre.leggauss(k+1)
        # self.int_x, self.int_w = np.polynomial.legendre.leggauss(k+2)
        self.int_n = len(self.int_w)
        self.psi, self.phi = [], []
        for i in range(k):
            p = np.polynomial.legendre.Legendre.basis(deg=i, domain=[-1, 1])
            self.psi.append(p)
            q = p.integ()
            self.phi.append(q-q(-1))
            assert self.phi[-1].deriv() == self.psi[-1]
        if plotbasis:
            import matplotlib.pyplot as plt
            t = np.linspace(-1,1)
            fig = plt.figure(figsize=plt.figaspect(2))
            ax = plt.subplot(211)
            for i in range(len(self.phi)):
                plt.plot(t, self.psi[i](t), '-', label=r"$\psi_{:2d}$".format(i))
            plt.legend()
            plt.grid()
            ax = plt.subplot(212)
            for i in range(len(self.phi)):
                plt.plot(t, self.phi[i](t), '-', label=r"$\phi_{:2d}$".format(i))
            plt.legend()
            plt.grid()
            plt.show()
        self.M = np.zeros(shape=(k,k))
        self.T = np.zeros(shape=(k,k))
        self.int_phiphi = np.zeros(shape=(k,k))
        self.int_psipsi = np.zeros(shape=(k,k))
        for i in range(k):
            for j in range(k):
                for ii in range(self.int_n):
                    self.M[i,j] += self.int_w[ii]* self.psi[i](self.int_x[ii])* self.phi[j](self.int_x[ii])
                    self.T[i, j] += self.int_w[ii] * self.psi[i](self.int_x[ii]) * self.psi[j](self.int_x[ii])
                    self.int_phiphi[i, j] += self.int_w[ii] * self.phi[i](self.int_x[ii]) * self.phi[j](self.int_x[ii])
                    self.int_psipsi[i, j] += self.int_w[ii] * self.psi[i](self.int_x[ii]) * self.psi[j](self.int_x[ii])
        self.diagT = np.diagonal(self.T)[1:]
        self.coefM = np.diagonal(self.M, offset=-1)
        if plotbasis:
            i = np.arange(1,k)
            assert np.allclose(self.diagT, 2/(2*i+1))
            assert np.allclose(self.coefM, 2/(4*i**2-1))
            # print(f"{self.diagT=}")
            # print(2/(2*i+1))
            # print(f"{self.coefM=}")
            # print(2/(4*i**2-1))
            sys.exit(1)
        self.int_psi_weights = np.empty(shape=(len(self.psi),self.int_n))
        self.int_psi = np.empty(shape=(len(self.psi),self.int_n))
        self.int_psik = np.empty(shape=(self.int_n))
        self.int_phi = np.empty(shape=(len(self.psi),self.int_n))
        for i in range(k):
            for ii in range(self.int_n):
                self.int_psi_weights[i,ii] = self.psi[i](self.int_x[ii])*self.int_w[ii]
                self.int_psi[i, ii] = self.psi[i](self.int_x[ii])
                self.int_phi[i, ii] = self.phi[i](self.int_x[ii])
        p = np.polynomial.legendre.Legendre.basis(deg=k, domain=[-1, 1])
        self.int_psik = p(self.int_x)
        self.int_phik2 = np.sum((self.phi[-1](self.int_x))**2*self.int_w)
        self.psi_mid = np.array([self.psi[ik](0) for ik in range(self.k)])
        self.phi_mid = np.array([self.phi[ik](0) for ik in range(self.k)])
#------------------------------------------------------------------
    def run_forward(self, t, app, linearization=None):
        app.type = 'numpy'
        if linearization is not None:
            utilde_node, utilde_coef = linearization
        u_ic = app.u0
        nt = t.shape[0]
        dim = 1 if type(u_ic) == float else len(u_ic)
        apphasl = hasattr(app,'l')
        dt = t[1:] - t[:-1]
        if apphasl:
            tm = 0.5*(t[1:]+t[:-1])
            lint = np.asarray(app.l(tm+0.5*self.int_x[:,np.newaxis]*dt)).T
        if linearization is not None:
            uint = utilde_node[:-1, np.newaxis] + np.einsum('ikp,kl->ilp', utilde_coef, self.int_phi)
            lintf = np.asarray(app.f(uint.T)).T.reshape(uint.shape)
        u_node = np.empty(shape=(nt, dim))
        u_coef = np.empty(shape=(nt-1, self.k, dim), dtype=u_node.dtype)
        bloc = np.empty(shape=(self.k,dim), dtype=u_node.dtype)
        Aloc = np.zeros((self.k,self.k, dim, dim), dtype=u_node.dtype)
        if hasattr(app, 'M'):
            M = app.M
        else:
            M = np.eye(dim)
        u_node[0] = u_ic
        for it in range(nt-1):
            dt = t[it+1]-t[it]
            assert(dt>0)
            bloc.fill(0)
            if linearization is None:
                utilde = u_node[it]
                if hasattr(app, 'f'):
                    f0 = np.asarray(app.f(utilde), dtype=u_node.dtype)
                else:
                    f0 = np.zeros_like(utilde)
                bloc[0] = dt * f0
            else:
                utilde = 0.5*(utilde_node[it]+utilde_node[it+1])
                bloc += 0.5*dt*np.einsum('jk,lj->lk', lintf[it], self.int_psi_weights)
            if hasattr(app, 'df'):
                A0 = np.asarray(app.df(utilde), dtype=u_node.dtype).reshape(dim, dim)
            else:
                A0 = np.zeros(shape=(dim,dim))
            if linearization is not None:
                bloc[0] += dt*A0@u_node[it]
            Aloc[0, 0] = 2*M - dt*A0
            for ik in range(1,self.k):
                Aloc[ik, ik] = self.diagT[ik - 1] * M
                Aloc[ik - 1, ik] = 0.5 * dt * self.coefM[ik - 1] * A0
                Aloc[ik, ik - 1] = -0.5 * dt * self.coefM[ik - 1] * A0
            if self.alpha:
                Aloc[self.k-1, self.k - 1] -= self.alpha*dt*A0
            if apphasl:
                bloc += 0.5*dt*np.einsum('jk,lj->lk', lint[it], self.int_psi_weights)
            usol = np.linalg.solve(Aloc.swapaxes(1,2).reshape(self.k*dim,self.k*dim), bloc.flat).reshape((self.k,dim))
            u_coef[it] = usol
            u_node[it + 1] = 2*usol[0] + u_node[it]
        return u_node.swapaxes(1,0), u_coef.swapaxes(-1,0)
# ------------------------------------------------------------------
    def interpolate(self, t, u_ap, mean=False):
        u_node, u_coef = u_ap
        if not mean: return u_node
        return u_node, u_node[:,:-1]+np.einsum('ijk,j', u_coef, self.phi_mid)
        # return u_node, u_node[:-1] + np.einsum('ijk,j', u_coef, self.phi_mid)
#------------------------------------------------------------------
    def compute_error(self, t, sol_ex, dsol_ex, u_ap):
        u_node, u_coef = u_ap
        dt = (t[1:]-t[:-1])
        tm = 0.5*(t[1:]+t[:-1])
        solt = np.asarray(sol_ex(t), dtype=u_node.dtype).T
        dsoltm = np.asarray(dsol_ex(tm), dtype=u_node.dtype).T
        udder = np.einsum('ijk,ij->ik', u_coef, self.psi_mid*2/dt[:,np.newaxis])
        e1 = np.fabs(solt-u_node)
        errfct = {'err_node': e1}
        errfct['err_der'] = np.fabs(dsoltm-udder)
        err = {}
        uder2 = np.asarray(dsol_ex(tm + 0.5 * dt * self.int_x[:,np.newaxis]), dtype=u_node.dtype).T
        uint2 = np.asarray(sol_ex(tm + 0.5 * dt * self.int_x[:,np.newaxis]), dtype=u_node.dtype).T
        uder_ap2 = 2*np.einsum('ijk,jl,i->ilk', u_coef, self.int_psi, 1/dt)
        uint_ap2 = u_node[:-1,np.newaxis] + np.einsum('ijk,jl->ilk', u_coef, self.int_phi)
        err['H1'] = 0.5*np.einsum('ilk,l,ik', (uder2-uder_ap2)**2, self.int_w, dt[:,np.newaxis])
        err['L2'] = 0.5*np.einsum('ilk,l,ik', (uint2-uint_ap2)**2, self.int_w, dt[:,np.newaxis])
        err['H1'] = np.sqrt(err['H1'])
        err['L2'] = np.sqrt(err['L2'])
        err['L2_nod'] = np.sqrt(0.5*np.sum( e1[1:]**2*dt[:,np.newaxis] + e1[:-1]**2*dt[:,np.newaxis], axis=0))
        err['max_nod'] = np.amax( np.fabs(e1), axis=0)
        em = np.asarray(sol_ex(tm), dtype=u_node.dtype).T - u_node[:-1]-np.einsum('ijk,j', u_coef, self.phi_mid)
        err['max_mid'] = np.amax( np.fabs(em), axis=0)
        return errfct, err
#------------------------------------------------------------------
    def estimator(self, t, u_ap, app):
        u_node, u_coef = u_ap
        apphasl = hasattr(app,'l')
        nt, dim = u_node.shape
        dt = t[1:] - t[:-1]
        tm = 0.5 * (t[1:] + t[:-1])
        df0 = np.array([np.sum( (app.df(u_node[it])@u_coef[it,-1])**2) for it in range(nt-1)])
        estap = 0.5 * df0*dt*self.int_phik2
        if apphasl:
            lint = np.asarray(app.l(tm+0.5*self.int_x[:,np.newaxis]*dt)).T
            lintk = np.einsum('ilk,l,l', lint, self.int_w, self.int_psik)
            estap += 0.5 * np.einsum('ik,ik->i', lintk ** 2, dt[:, np.newaxis])
        df = np.array([np.sum((np.array(app.df(u_node[it])) - np.array(app.df(u_node[it+1]))) ** 2) for it in range(nt - 1)])
        u2 = np.einsum('ikp,ilp,kl->i', u_coef, u_coef, self.int_phiphi)
        estnl = 0.5*dt*df*u2
        return {"nl": estnl, "ap": estap}, {'sum': np.sqrt(np.sum(estnl+estap))}
#------------------------------------------------------------------
if __name__ == "__main__":
    import ode_examples
    import matplotlib.pyplot as plt

    cgp = CgK(k=2)
    app = ode_examples.Logistic()
    t = np.linspace(app.t_begin, app.t_end, 10)
    u_node, u_coef = cgp.run_forward(t, app)
    u_true = app.solution(t)
    plt.subplot(121)
    plt.plot(t, u_node.T, label='Approximation')
    plt.plot(t, u_true, '--', label='Solution')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title(f"{app.name}")
    plt.grid()
    plt.subplot(122)
    plt.xlabel("t")
    plt.ylabel("e(t)")
    plt.plot(t, u_true - u_node[0,:])
    plt.grid()
    plt.show()