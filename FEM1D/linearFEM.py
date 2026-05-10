import numpy as np
from scipy import sparse
from types import SimpleNamespace
import bernstein, linearSolver


#-------------------------------------------------------------
class LinearElliptic:
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

    def make_fe_vector(self, mesh):
        nx, k, ncomp = mesh.shape[0], self.korder, self.app.ncomp
        bubbles = np.zeros(shape=(nx - 1, k - 1, ncomp)) if k > 1 else None
        return SimpleNamespace(p1=np.zeros(shape=(nx, ncomp)), bubbles=bubbles)

    # -------------------------------------------------------------
    def make_local_fe_matrix(self, mesh):
        nx, k, ncomp = mesh.shape[0], self.korder, self.app.ncomp
        nb = max(k - 1, 0) * ncomp
        nv = 2 * ncomp
        return SimpleNamespace(
            App=np.zeros(shape=(nx - 1, nv, nv)),
            Apb=np.zeros(shape=(nx - 1, nv, nb)) if k > 1 else None,
            Abp=np.zeros(shape=(nx - 1, nb, nv)) if k > 1 else None,
            Abb=np.zeros(shape=(nx - 1, nb, nb)) if k > 1 else None,
        )

    # -------------------------------------------------------------
    def __init__(self, korder, app):
        self.korder, self.app = korder, app
        self.integration = {'coeff': bernstein.reference_integration(2 * korder + 2, korder),
                            'rhs': bernstein.reference_integration(2*korder + 2, korder),
                            # 'est': bernstein.reference_integration(2*korder + 2, korder),
                            'error': bernstein.reference_integration(2*korder + 3, korder)}
        self.linear_solver = linearSolver.CondensedLinearSolver()

    def make_laplace_local_scalar(self, mesh):
        integ = self.integration["coeff"]
        dx = mesh[1:] - mesh[:-1]
        Aref = np.einsum("iq,jq,q->ij", integ.dphi, integ.dphi, integ.w)
        return np.einsum("e,ij->eij", 1.0 / dx, Aref)

    def matrix_p1_fast(self, mesh):
        nx, ncomp = mesh.shape[0], self.app.ncomp
        h = mesh[1:] - mesh[:-1]
        main = np.zeros(nx)
        off = np.zeros(nx - 1)
        main[:-1] += 1.0 / h
        main[1:] += 1.0 / h
        off[:] = -1.0 / h
        A1 = sparse.diags(
            diagonals=[off, main, off],
            offsets=[-1, 0, 1],
            shape=(nx, nx),
            format="csc"
        )
        if ncomp == 1:
            return A1
        return sparse.kron(A1, sparse.eye(ncomp, format="csc"), format="csc")
    def local_coeffs(self, mesh, uh):
        ne, k, ncomp = mesh.shape[0] - 1, self.korder, self.app.ncomp
        uloc = np.zeros((ne, k + 1, ncomp))
        uloc[:, 0, :] = uh.p1[:-1, :]
        uloc[:, k, :] = uh.p1[1:, :]
        if k > 1:
            uloc[:, 1:k, :] = uh.bubbles
        return uloc
    def compute_error(self, mesh, uh):
        ne, k, ncomp = mesh.shape[0] - 1, self.korder, self.app.ncomp
        integ = self.integration['error']
        # nint = integ.n
        xm, dx = mesh[:-1],  (mesh[1:] - mesh[:-1])
        x_rhs = xm[:, None] + dx[:, None] * integ.xi[None, :]  # shape (nx-1, int_n)
        uloc = self.local_coeffs(mesh, uh)
        # print(f"{uloc.shape=} {integ.phi.shape=}")
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
    def compute_estimator(self, mesh, uh):
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
        f_all = self.app.f_vec(x_rhs, u_vals)
        if ncomp == 1 and f_all.ndim == 2:
            f_all = f_all[:, :, None]
        res = f_all + d2u_dx2

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
        return np.sqrt(np.sum(eta_vol_sq))/np.pi/k, eta_vol_sq


    def rhs(self, mesh, uh):
        nx, k, ncomp, app = mesh.shape[0], self.korder, self.app.ncomp, self.app
        integ = self.integration['rhs']
        nint = integ.n
        xm, dx = mesh[:-1],  (mesh[1:] - mesh[:-1])
        x_rhs = xm[:, None] + dx[:, None] * integ.xi[None, :]  # shape (nx-1, int_n)
        uloc = self.local_coeffs(mesh, uh)
        # print(f"{uloc.shape=} {integ.phi.shape=}")
        u_vals = np.einsum('ijc,jq->iqc', uloc, integ.phi)
        # --- Evaluate f(x,u) at all quadrature points ---
        # We collect all values and stack to (nx-1, int_n, ncomp)
        f_all = app.f_vec(x_rhs, u_vals)
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
        rhs.rhs_loc = rhs_loc
        return rhs
    def matrix(self, mesh, uh):
        nx, k, ncomp = mesh.shape[0], self.korder, self.app.ncomp
        if k == 1:
            Aglob = self.matrix_p1_fast(mesh)
            return SimpleNamespace(local=None, global_p1=Aglob.tocsc())

        ne = nx - 1
        Alocal = self.make_local_fe_matrix(mesh)
        Aglob = sparse.lil_matrix((nx * ncomp, nx * ncomp))
        # lap.shape == (ne, k+1, k+1)
        lap = self.make_laplace_local_scalar(mesh)
        iv = [0, k]
        ib = list(range(1, k))
        for e in range(ne):
            # scalar local matrix
            Ae = lap[e]
            # component-expanded local matrix
            # Avec shape: (k+1, ncomp, k+1, ncomp)
            Avec = np.einsum("ij,ab->iajb", Ae, np.eye(ncomp))
            # vertex/bubble split
            App = Avec[iv, :, :, :][:, :, iv, :].reshape(2 * ncomp, 2 * ncomp)
            Alocal.App[e, :, :] = App
            if k > 1:
                Apb = Avec[iv, :, :, :][:, :, ib, :].reshape(2 * ncomp, (k - 1) * ncomp)
                Abp = Avec[ib, :, :, :][:, :, iv, :].reshape((k - 1) * ncomp, 2 * ncomp)
                Abb = Avec[ib, :, :, :][:, :, ib, :].reshape((k - 1) * ncomp, (k - 1) * ncomp)
                Alocal.Apb[e, :, :] = Apb
                Alocal.Abp[e, :, :] = Abp
                Alocal.Abb[e, :, :] = Abb
            # assemble only App globally for now
            gdofs = []
            for node in [e, e + 1]:
                for c in range(ncomp):
                    gdofs.append(node * ncomp + c)
            Aglob[np.ix_(gdofs, gdofs)] += App
        return SimpleNamespace(local=Alocal, global_p1=Aglob.tocsc())

    def dirichlet(self, mesh, Ah, fh):
        nx, k, ncomp = mesh.shape[0], self.korder, self.app.ncomp
        Aglobal, Alocal = Ah.global_p1, Ah.local
        # zero RHS boundary values
        fh.p1[0, :] = 0.0
        fh.p1[-1, :] = 0.0
        bc = []
        for c in range(ncomp):
            bc.append(c)
            bc.append((nx - 1) * ncomp + c)
        A = Aglobal.tolil()
        for i in bc:
            A[i, :] = 0.0
            # A[:, i] = 0.0
            A[i, i] = 1.0
        if Alocal is not None:
            # left boundary vertex rows
            for c in range(ncomp):
                r = c  # local left vertex component
                Alocal.Apb[0, r, :] = 0.0
                fh.p1[0, c] = 0.0
            # right boundary vertex rows
            for c in range(ncomp):
                r = ncomp + c  # local right vertex component
                Alocal.Apb[-1, r, :] = 0.0
                fh.p1[-1, c] = 0.0
        return SimpleNamespace(local=Ah.local, global_p1=A.tocsc()), fh
    def solve_linear(self, mesh, Ah, fh):
        return self.linear_solver.solve(Ah, fh)

    def solve(self, mesh, uh=None):
        if uh is None:
            uh = self.make_fe_vector(mesh)
        fh = self.rhs(mesh, uh)
        Ah = self.matrix(mesh, uh)
        Ah, fh = self.dirichlet(mesh, Ah, fh)
        return self.solve_linear(mesh, Ah, fh)

if __name__ == '__main__':
    import elliptic_examples
    from Utility import mesh1d, plotting, printer
    import matplotlib.pyplot as plt
    import time

    def check_error(app, korder=2, niter=10):
        ns, errL2, errH1, etas, times = [], [], [], [], []
        for iter in range(niter):
            n = 10*2**iter
            mesh = mesh1d.mesh(app.x0, app.x1, n=n, type='random')
            solver = LinearElliptic(korder=korder, app=app)
            t0 = time.time()
            uh = solver.solve(mesh)
            eL2, eH1 = solver.compute_error(mesh, uh)
            eta, eta_sq = solver.compute_estimator(mesh, uh)
            ns.append(n), errL2.append(eL2), errH1.append(eH1), etas.append(eta), times.append(time.time()-t0)
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
        plot_dict = {f"Errors {app.name} k={korder}":  pdict}
        plotting.plot_error_curves(plot_dict)
        plt.show()


    def plot_solution(app, korder=2, nmesh=10):
        mesh = mesh1d.mesh(app.x0, app.x1, n=nmesh, type='random')
        solver = LinearElliptic(korder=korder, app=app)
        #solver.plot_basis()
        t0 = time.time()
        uh = solver.solve(mesh)
        err = solver.compute_error(uh)
        print(f"time: {time.time() - t0} error: {err}")
        pd = {app.name: {}}
        pd[app.name]['x'] = mesh
        pd[app.name]['y'] = {'uh': uh.p1}
        pd[app.name]['kwargs'] = {'uh': {'marker': 'o'}}
        if hasattr(app, 'solution'):
            pd[app.name]['y']['u'] = app.solution(mesh)
            pd[app.name]['y']['err'] = np.fabs(app.solution(mesh)[:,None]-uh.p1)
        plotting.plot_solutions(pd)
        plt.show()

    # app = elliptic_examples.Poisson()
    # app = elliptic_examples.SmoothPoisson()
    app = elliptic_examples.OscillatoryPoisson(omega=5, alpha=3.0)
    check_error(app, niter=8, korder=3)
    # plot_solution(app, nmesh=20, korder=2)

