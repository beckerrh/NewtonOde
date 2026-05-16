import linearFEM
from Utility import mesh1d, plotting, logger, timer
from types import SimpleNamespace
from Newton import armijo


class NewtonDriver:
    def __init__(self, app, n0, korder, mesh=None):
        self.app = app
        self.mesh = mesh if mesh is not None else mesh1d.mesh(app.x0, app.x1, n=n0, type='uniform')
        self.solver = linearFEM.LinearElliptic(korder=korder, app=app)
        self.lam0 = 0.9
        self.C0 = 100.0
        self.beta = 0.5
        self.theta = 0.9
        types = {'it': 'd', 'zeta': 'e', 'N': 'd:7', '|p|':'e', 'eta': 'e', 'mer':'e', 'mu':'e'}
        self.logger_mesh = logger.Logger(types=types, name="mesh")
        self.globalization = armijo.ArmijoGlobalization(
            maxiter=20,
            omega=0.75,
            c=1e-4,
            relative_decrease=True
        )
        self.timer = timer.Timer()

    def attach_logger(self, logger):
        self.logger_newton = logger
        self.logger_newton.add_types({'eta':'e', '|F_h|':'e', 'N': 'd:7', 'meshiter': 'd:3'})
        self.logger_newton.update(N=len(self.mesh))
    def initial_guess(self):
        return self.solver.make_fe_vector(self.mesh)
    def add_update(self, x, alpha, p):
        return self.solver.add_update(x, alpha, p)
    def evaluate(self, x):
        with self.timer("residual"):
            res = self.solver.compute_residual(self.mesh, x)
        resn = self.solver.normHminus1(self.mesh, res)
        with self.timer("estimator"):
            result = self.solver.compute_estimator_nonlinear(self.mesh, x)
        meritvalue = result.eta_global+self.beta*resn
        self.logger_newton.update(eta=result.eta_global)
        self.logger_newton.values["|F_h|"] = resn
        xnorm = self.solver.normH1(self.mesh, x)
        return SimpleNamespace(residual=res, meritvalue=meritvalue, norm_X=xnorm)

    def compute_newton_step(self,  x, state, iterdata):
        aimed = self.lam0*state.meritvalue
        n_mesh_iter=100
        for iter_mesh in range(n_mesh_iter):
            with self.timer("compute_residual"):
                r = self.solver.compute_residual(self.mesh, x)
            resn = self.solver.normHminus1(self.mesh, r)
            with self.timer("solve_linearized"):
                p = self.solver.solve_linearized(self.mesh, x, r)
            with self.timer("estimator"):
                resultnl = self.solver.compute_estimator_nonlinear(self.mesh, x)
                result = self.solver.compute_estimator(self.mesh, x, p)
            pnorm = self.solver.normH1(self.mesh, p)
            if iter_mesh == 0:
                self.logger_mesh.print_names()
            self.logger_mesh.update(it=iter_mesh, zeta=result.deta_global,
                                     N=len(self.mesh), eta=resultnl.eta_global,
                                     mer=resultnl.eta_global+self.beta*resn)
            self.logger_mesh.values['|p|'] = pnorm
            self.logger_mesh.print()
            # print(f"\t  {info.iter} \t {iter_mesh} \t {aimed} \t {result.eta_global} \t {result.zeta_global}\t {len(self.mesh)}\t {pnorm}")
            if result.deta_global <= aimed or result.deta_global < iterdata.tol_aimed:
                self.logger_newton.update(N=len(self.mesh), meshiter=iter_mesh)
                return SimpleNamespace(dx=p, dx_norm=pnorm, x=x, success=True)

            mesh_new, refinfo = mesh1d.adapt_mesh(self.mesh, result.deta, theta=self.theta)
            x = self.solver.interpolate_to_new_mesh(self.mesh, x, mesh_new, refinfo)
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



#------------------------------------------------------------------
if __name__ == "__main__":
    from Newton import newton, newtondata
    import elliptic_examples
    # app = elliptic_examples.OscillatoryPoisson(omega=9, alpha=13.0)
    app = elliptic_examples.LinearSystem3()

    sdata = newtondata.StoppingParamaters(maxiter=12, rtol=1e-10)
    driver = NewtonDriver(app, n0=10, korder=4)
    x0 = driver.initial_guess()
    newton = newton.Newton(nd = driver, verbose=2, sdata=sdata)
    xs, info, logger = newton.solve(x0)

    print('---- time ---')
    print(driver.timer.summary(),'\n')
    print(driver.solver.timer.summary(),'\n')
    print(driver.solver.linear_solver.timer.summary())

    logger.print_history()


