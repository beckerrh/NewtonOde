import linearFEM
from Utility import mesh1d, plotting, logger, timer
from types import SimpleNamespace
from Newton import armijo
import numpy as np


class NewtonDriver:
    def __init__(self, app, n0, korder, mesh=None, verbose=1):
        self.app = app
        self.mesh = mesh if mesh is not None else mesh1d.mesh(app.x0, app.x1, n=n0, type='uniform')
        self.verbose = verbose
        self.solver = linearFEM.LinearElliptic(korder=korder, app=app)
        self.beta = 0.9
        self.theta = 0.9
        types = {'it': 'd', 'deta': 'e', 'N': 'd:7', '|p|':'e'}
        if self.verbose > 1:
            types.update({'eta': 'e', '|F_h|':'e', 'mer':'e', 'beta':'e'})
        self.logger_mesh = logger.Logger(types=types, name="mesh")
        self.timer = timer.Timer()
        self.merits, self.etas, self.errs, self.ns = [], [], [], []
        self.globalization = armijo.ArmijoGlobalization(
            omega=0.5,
            c=1e-4,
            maxiter=20,
            verbose=False, relative_decrease=True
        )

    def attach_logger(self, logger):
        self.logger_newton = logger
        types = {'eta':'e', '|F_h|':'e', 'N': 'd:7', 'meshiter': 'd:3'}
        self.logger_newton.add_types(types)
        self.logger_newton.update(N=len(self.mesh))
    def initial_guess(self):
        uh = self.solver.make_fe_vector(self.mesh)
        uh.p1[0, :] = self.app.uL
        uh.p1[-1, :] = self.app.uR
        return uh
    def add_update(self, x, alpha, p):
        return self.solver.add_update(x, alpha, p)
    def evaluate(self, x):
        with self.timer("residual"):
            res = self.solver.compute_residual(self.mesh, x)
        resn = self.solver.normHminus1(self.mesh, res)
        with self.timer("estimator"):
            result = self.solver.compute_estimator_nonlinear(self.mesh, x)
        self.eta_new = result.eta_global
        merit2 = result.eta_global ** 2 + self.beta * resn ** 2
        meritvalue = np.sqrt(merit2)
        self.logger_newton.update(eta=result.eta_global)
        self.logger_newton.values["|F_h|"] = resn
        xnorm = self.solver.normH1(self.mesh, x)
        return SimpleNamespace(residual=res, meritvalue=meritvalue, norm_X=xnorm)

    def compute_newton_step(self,  x, state, info):
        aimed = info.tol_linear_abs
        n_mesh_iter=100
        for iter_mesh in range(n_mesh_iter):
            if iter_mesh == 0:
                self.logger_mesh.print_names(add=f"{aimed=:.4e}")
                prev = None
            with self.timer("compute_residual"):
                r = self.solver.compute_residual(self.mesh, x)
            with self.timer("solve_linearized"):
                p = self.solver.solve_linearized(self.mesh, x, r)
            pnorm = self.solver.normH1(self.mesh, p)
            with self.timer("estimator"):
                result = self.solver.compute_estimator_linearized(self.mesh, x, p)
            self.logger_mesh.update(it=iter_mesh, deta=result.deta_global, N=len(self.mesh))
            self.logger_mesh.values['|p|'] = pnorm

            if self.verbose >1:
                resn = self.solver.normHminus1(self.mesh, r)
                with self.timer("estimator"):
                    resultnl = self.solver.compute_estimator_nonlinear(self.mesh, x)
                eta_new = resultnl.eta_global
                # self.eta_new = eta_new
                if prev is None:
                    beta2 = np.nan
                else:
                    eta_old, resn_old = prev
                    den = resn ** 2 - resn_old ** 2
                    beta2 = (eta_old ** 2 - eta_new ** 2) / den if den > 0 else np.nan
                prev = (eta_new, resn)
                self.logger_mesh.values['beta'] = beta2
                self.logger_mesh.values['mer'] = np.sqrt(resultnl.eta_global ** 2 + self.beta * resn ** 2)
                self.logger_mesh.values['eta'] = resultnl.eta_global
                self.logger_mesh.values['|F_h|'] = resn

            self.logger_mesh.print()
            if result.deta_global <= aimed:
                self.logger_newton.update(N=len(self.mesh), meshiter=iter_mesh)
                return SimpleNamespace(dx=p, dx_norm=pnorm, x=x, success=True)

            mesh_new, refinfo = mesh1d.adapt_mesh(self.mesh, result.deta, theta=self.theta)
            x = self.solver.interpolate_to_new_mesh(self.mesh, x, mesh_new, refinfo)
            x.p1[0, :] = self.app.uL
            x.p1[-1, :] = self.app.uR
            self.mesh = mesh_new
        return SimpleNamespace(dx=p, dx_norm=pnorm, x=x, success=False)
    def call_back(self, iterdata, accepted):
        import matplotlib.pyplot as plt
        uh = accepted.x.p1
        name = self.app.name
        plot_dict = {name: {'x': self.mesh}}
        plot_dict[name]['y'] = {'uh': uh}
        plot_dict = plotting.add_mesh1d(plot_dict, self.mesh)
        title = f"{name} =={accepted.success}== Iter {iterdata.iter}"
        plotting.plot_solutions(plot_dict, title=title)
        plt.show()
        # xplot, uplot = self.solver.eval_fe_on_grid(self.mesh, accepted.x, nsub=100)
        # uex = app.solution(xplot)
        # plt.plot(xplot, uplot[:, 0], label="FE full")
        # plt.plot(xplot, uex[:, 0], "--", label="exact")
        # plt.xlim(0, 0.15)
        # plt.legend()
        # plt.grid()
        # plt.show()
        if iterdata.success:
            self.ns.append(len(self.mesh)*self.solver.korder)
            self.etas.append(self.eta_new)
            if hasattr(self.app,'solution'):
                eL2, eH1 = self.solver.compute_error(self.mesh, accepted.x)
                self.errs.append(eH1)
    def plot_eta(self):
        import matplotlib.pyplot as plt
        ns = np.array(self.ns)
        errs = np.array(self.errs)
        etas = np.array(self.etas)
        pdict = {'x': ns, 'y': {'eta': etas}}
        if hasattr(self.app, 'solution'):
            pdict['y']['err'] = errs
        plot_dict = {f"Errors {app.name} k={self.solver.korder}":  pdict}
        plotting.plot_error_curves(plot_dict, fixed_order=self.solver.korder)
        plt.show()



#------------------------------------------------------------------
if __name__ == "__main__":
    from Newton import newton, newtondata
    import elliptic_examples
    # app = elliptic_examples.OscillatoryPoisson(omega=9, alpha=13.0)
    # app = elliptic_examples.ExampleSystem(rho=2)
    # app = elliptic_examples.PotentialReactionSystem(rho=10)
    # app = elliptic_examples.ScalarCubic(A=1, eps=0.01, rho=10000)
    # app = elliptic_examples.SimpleMonotone2()
    app = elliptic_examples.LayeredMonotone2()

    korder, rtol = 2, 1e-10
    # korder, rtol = 1, 1e-8
    sdata = newtondata.StoppingParamaters(maxiter=50, rtol=rtol, forcing_kappa=0.5)
    driver = NewtonDriver(app, n0=11, korder=korder)
    x0 = driver.initial_guess()
    newton = newton.Newton(nd = driver, verbose=2, sdata=sdata)
    xs, info, logger = newton.solve(x0)

    if not info.success:
        print(info.success,info.failure)
    else:
        print('---- time ---')
        print(driver.timer.summary(),'\n')
        print(driver.solver.timer.summary(),'\n')
        print(driver.solver.linear_solver.timer.summary())
        driver.plot_eta()
        logger.print_history()


