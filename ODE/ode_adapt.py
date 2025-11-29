import numpy as np
from ode_solver import ODE_Legendre
from Utility import mesh1d, plotting, printer
import ode_examples
import matplotlib.pyplot as plt
from types import SimpleNamespace

#==================================================================
def plot_all(**kwargs):
    mesh = kwargs.get("mesh")
    app = kwargs.get("app")
    x = kwargs.get("x")
    solver = kwargs.get("solver")
    t_mp, u_mp = solver.interpolate_midpoint(mesh, x)
    pd = {app.name: {}, "Mesh": {}}
    pd[app.name]['x'] = t_mp
    pd[app.name]['y'] = {'app': u_mp}
    if hasattr(app, "solution"):
        u_true = app.solution(t_mp).squeeze()
        pd[app.name]['y']['sol'] = u_true
    # pd[app.name]['kwargs'] = {'app': {'marker': 'o'}}
    pd["Mesh"]['x'] = mesh
    hinv = 1.0 / (mesh[1:] - mesh[:-1])
    h0 = np.min(hinv)
    hinv /= h0
    pd["Mesh"]['y'] = {rf'$\frac{{{1/h0:.2e}}}{{h}}$': hinv}
    pd["Mesh"]['type'] = 'step'
    cand_cell = ['zeta', 'eta', 'mu', 'err']
    cell_dict={}
    for c in cand_cell:
        if c in kwargs:
            cell_dict[c] = kwargs[c]
    if len(cell_dict):
        pd["Estimator"] = {}
        pd["Estimator"]['x'] = t_mp
        pd["Estimator"]['y'] = {}
    for c,v in cell_dict.items():
        pd["Estimator"]['y'][c] = v
    # pd["Estimator"]['y'] = {'zeta_cell': zeta_cell, 'eta': eta_cell, 'err': err_cell}
    title = kwargs.get("title", app.name)
    plotting.plot_solutions(pd, title=title)
    if hasattr(app, "plot"): app.plot(t_mp, u_mp.T)
    plt.show()


#==================================================================
class Newton_Ode():
    def __init__(self, app, k=0, mesh=None, n0=12):
        self.lam0 = 0.15
        self.C0 = 100.0
        self.beta = 1.0
        self.theta = 0.9
        self.solver = ODE_Legendre(k, ncomp=app.ncomp)
        self.app = app
        if not mesh:
            self.mesh = mesh1d.mesh(app.t_begin, app.t_end, n=n0)
        else:
            self.mesh = mesh
        types = {'it': 'd', 'zeta': 'e', 'N': 'd:7', '|p|':'e', 'eta': 'e', 'mer':'e'}
        self.printer_mesh = printer.Printer(types=types, name="mesh")
    def attach_printer(self, printer):
        self.printer_newton = printer
        self.printer_newton.add_types({'eta':'e', '|R|':'e', 'N': 'd:7'})
        self.printer_newton.update(N=len(self.mesh))
    def initial_guess(self):
        return self.solver.solve_semi_implicit(self.mesh, self.app)
        nt = self.mesh.shape[0]
        nbasis = len(self.solver.phi)
        ncomp = 1 if np.ndim(self.app.x0) == 0 else len(self.app.x0)
        # x0 = np.asarray(self.app.x0)
        x0all = np.zeros(shape=(nt-1, nbasis, ncomp))
        x0all[:,0] = x0
        # print(f"{x0=} {x0all=}")
        return x0all
    def add_update(self, x, step, p):
        # xnew = (x[0]+step*p[0], x[1]+step*p[1])
        return x + step*p
    # def update_rule(self, x, dx):
    #     xnew = self.add_update(x, 0.9, dx)
    #     return xnew, 0.9
    def computeMeritGrad(self, x, dx, result_merit):
        eta_grad = self.solver.estimator_merit_grad(self.mesh, self.app, x, dx)
        print(f"{eta_grad=} {result_merit.resn=}")
        return self.beta*eta_grad - result_merit.resn
    def computeMeritFunction(self, x):
        res = self.solver.compute_residual(self.mesh, self.app, x)
        resn = self.solver.norm_residual(self.mesh, res)
        result = self.solver.estimator(self.mesh, x, self.app)
        meritvalue = self.beta*result.eta_global+resn
        self.printer_newton.update(eta=result.eta_global)
        self.printer_newton.values["|R|"] = resn
        # print(f"computeResidual {x=}\n{res=}")
        # resnorm = np.linalg.norm(res[0])+np.linalg.norm(res[1])
        return SimpleNamespace(meritvalue=meritvalue,
                               x_norm=self.solver.norm_X(self.mesh, x),
                               resn=resn)
    def computeUpdate(self, r, x, info):
        aimed = self.lam0*info.resnorm[-1]
        n_mesh_iter=100
        print(f"{info.tol_aimed=}")
        for iter_mesh in range(n_mesh_iter):
            r = self.solver.compute_residual(self.mesh, self.app, x)
            resn = self.solver.norm_residual(self.mesh, r)
            p = self.solver.solve_linearized(self.mesh, self.app, x, r)
            result = self.solver.estimator(self.mesh, x, self.app, p)
            pnorm = self.solver.norm_X(self.mesh, p)
            if iter_mesh == 0:
                self.printer_mesh.print_names()
            self.printer_mesh.update(it=iter_mesh, zeta=result.zeta_global,
                                     N=len(self.mesh), eta=result.eta_global,
                                     mer=self.beta*result.eta_global+resn)
            self.printer_mesh.values['|p|'] = pnorm
            self.printer_mesh.print()

            # print(f"\t  {info.iter} \t {iter_mesh} \t {aimed} \t {result.eta_global} \t {result.zeta_global}\t {len(self.mesh)}\t {pnorm}")
            if result.zeta_global <= aimed*min(1.0, self.C0*pnorm) or iter_mesh == n_mesh_iter - 1:
                self.printer_newton.update(N=len(self.mesh))
                return SimpleNamespace(update=p, update_norm=pnorm, x=x)
            mesh_new, refinfo = mesh1d.adapt_mesh(self.mesh, result.zeta_cell, theta=self.theta)
            xnew = self.solver.interpolate(x, mesh_new, refinfo)
            x = xnew
            self.mesh = mesh_new
        assert None
    def call_back(self, **kwargs):
        x = kwargs["x"]
        iterdata = kwargs["iterdata"]
        plot_all(mesh=self.mesh, app=self.app, x=x, solver=self.solver,
                 title=f"{self.app.name} Iter {iterdata.iter}")

    def call_back_backtrack_failed(self, **kwargs):
        btresult = kwargs['btresult']
        dx = kwargs['dx']
        r = kwargs['r']
        if not btresult.success:
            print(f"{btresult.step=}")
        else:
            print(f"{kwargs['dxnorm']=} {kwargs['dxnorm_old']=}")
    def plot(self, x):
        pass
        # print(f"{x.shape=} {self.mesh.shape=}")




#==================================================================
def test_interpolation(niter=6):
    app = ode_examples.PolynomialIntegration(degree=9, ncomp=1)
    solver = ODE_Legendre(k=3)
    mesh = mesh1d.mesh(app.t_begin, app.t_end, n=4)
    for iter in range(niter):
        ucoeff = solver.run(mesh, app.x0, app.a_coef, app.b_coef)
        eta, eta_cell = solver.estimator(mesh, ucoeff, app.a_coef, app.b_coef)
        mesh_new, refinfo = mesh1d.adapt_mesh(mesh, eta_cell)
        ucoeff_new = solver.interpolate(ucoeff, mesh_new, refinfo)
        plt.title(f"Test Interpolation Iteration {iter}")
        solver.plot_dg(mesh, ucoeff, {'color': 'blue', 'ls':'--', 'marker':'X', 'label':'uold'})
        solver.plot_dg(mesh_new, ucoeff_new, {'color': 'red', 'ls':'--', 'marker':'x', 'label':'unew'})
        plt.grid()
        plt.legend()
        plt.show()
        mesh = mesh_new

#==================================================================
def adaptive_linear(app, solver, **kwargs):
    eta_rtol = kwargs.get("eta_rtol", None)
    niter = kwargs.get("niter", 30)
    plot = kwargs.get("plot", False)
    n0 = kwargs.get("n0", 11)
    semi_implicit = kwargs.get("semi_implicit", True)
    theta = kwargs.get("theta", 0.9)
    mesh = mesh1d.mesh(app.t_begin, app.t_end, n=n0)
    ns, errs_l2, etas = [], [], []
    has_solution = hasattr(app, 'solution')
    for iter in range(niter):
        if semi_implicit:
            x = solver.solve_semi_implicit(mesh, app)
        else:
            x = solver.solve_linear(mesh, app)
        est = solver.estimator(mesh, x, app)
        ns.append(mesh.shape[0])
        etas.append(est.eta_global)
        if has_solution:
            el2, err_cell = solver.compute_error(mesh, x, app.solution)
            errs_l2.append(el2)
            msg = f"{ns[-1]:7d} {errs_l2[-1]/etas[-1]:8.2f} {etas[-1]:12.2e}"
        else:
            msg = f"{ns[-1]:7d} {etas[-1]:12.2e}"
        if plot:
            kwargs = {'mesh':mesh, 'app':app, 'solver':solver, 'x':x, 'eta':np.sqrt(est.eta_cell), 'title':f"Iteration {iter} N={mesh.shape[0]}"}
            if has_solution: kwargs['err'] = np.sqrt(err_cell)
            plot_all(**kwargs)
        if eta_rtol is not None:
            if iter==0: eta_first = est.eta_global
            msg += f"{est.eta_global:12.2e} {eta_first*eta_rtol:12.2e}"
            if est.eta_global <= eta_rtol*eta_first:
                print(msg)
                result = SimpleNamespace(ns=np.array(ns), etas=np.array(etas), err_l2=np.array(errs_l2))
                return result
        print(msg)
        mesh_new, refinfo = mesh1d.adapt_mesh(mesh, est.eta_cell, theta=theta)
        x_new = solver.interpolate(x, mesh_new, refinfo)
        mesh = mesh_new
    result = SimpleNamespace(ns=np.array(ns), etas=np.array(etas))
    if has_solution: result.err_l2 = np.array(errs_l2)
    return result
def test_adaptive_linear(app, solver, **kwargs):
    res = adaptive_linear(app, solver, **kwargs)
    pdict = {'x': res.ns, 'y': {'eta': res.etas}}
    if hasattr(res, 'err_l2'): pdict['y']['err'] = res.err_l2
    plot_dict = {"Errors":  pdict}
    plotting.plot_error_curves(plot_dict)
    plt.show()
def compare_adaptive_linear(app, solver, **kwargs):
    thetas = [0.7, 0.9, 1.0]
    ns_all, errs_l2_all, etas_all = [], [], []
    for theta in thetas:
        kwargs["theta"] = theta
        ns, errs_l2, etas = adaptive_linear(app, solver, **kwargs)
        ns_all.append(ns)
        errs_l2_all.append(errs_l2)
        etas_all.append(etas)
    plot_dict = {}
    for theta, ns, errs_l2, etas in zip(thetas, ns_all, errs_l2_all, etas_all):
        plot_dict[rf"$\theta={theta}$"] = {'x': ns, 'y': {'L2': errs_l2, 'eta': etas}}
    plotting.plot_error_curves(plot_dict)
    plt.show()
    plot_dict = {"Etas":{}, "Errors":{}}
    ys_eta, ys_err = {}, {}
    for theta, errs_l2, etas in zip(thetas, errs_l2_all, etas_all):
        ys_eta[rf"$\theta={theta}$"] = etas
        ys_err[rf"$\theta={theta}$"] = errs_l2
    plot_dict["Etas"]['x'] = ns_all
    plot_dict["Etas"]['y'] = ys_eta
    plot_dict["Errors"]['x'] = ns_all
    plot_dict["Errors"]['y'] = ys_err
    plotting.plot_error_curves(plot_dict)
    plt.show()



#==================================================================
def test_adaptive(app, solver, niter=6, plot=True):
    pass

#------------------------------------------------------------------
if __name__ == "__main__":
    # todo = 'semi_implicit'
    todo = 'nonlinear'
    # todo = 'linear'
    np.random.seed(12)
    if todo == 'interpolation':
        test_interpolation()
    elif todo == 'linear':
        # app = ode_examples.LinearIntegration()
        # app = ode_examples.PolynomialIntegration(degree=8, ncomp=2)
        # app = ode_examples.Exponential(lam=10.2)
        # app = ode_examples.ExponentialJordan(lam=2.2)
        # app = ode_examples.TimeDependentShear()
        app = ode_examples.TimeDependentRotation()
        solver = ODE_Legendre(k=2, ncomp=app.ncomp)
        # app = ode_examples.RotScaleForce()
        test_adaptive_linear(app, solver, semi_implicit=False, eta_rtol=1e-6, plot=True)
    elif todo == 'semi_implicit':
        # app = ode_examples.TimeDependentRotation()
        # app = ode_examples.LinearSingular()
        # app = ode_examples.ArctanJump(eps=0.05)
        # app = ode_examples.Logistic(t_end=10)
        app = ode_examples.Lorenz(t_end=18, x0=[1.0, 1.0, 1.0])
        app = ode_examples.LinearPBInstability(t_end=10.0)
        app = ode_examples.Robertson(t_end=100.0) # needs very fine initial mesh
        solver = ODE_Legendre(k=0, ncomp=app.ncomp, est_degree=20, error_degree=20)
        test_adaptive_linear(app, solver, semi_implicit=True, eta_rtol=1e-6, plot=True, n0=500, niter=1)
        # compare_adaptive_linear(app, solver, semi_implicit=True, eta_rtol=1e-6, niter=100, n0=20)
    else:
        from Newton import newton, newtondata
        # app = ode_examples.PolynomialIntegration(degree=8, ncomp=2)
        # app = ode_examples.Exponential(lam=0.2)
        # app = ode_examples.ExponentialJordan(lam=2.2)
        # app = ode_examples.Logistic()
        # app = ode_examples.Pendulum(t_end=13)
        app = ode_examples.DoublePendulum(t_end=25)
        # app = ode_examples.VanDerPol(t_end=32.0)
        # app = ode_examples.Robertson(t_end=1.0)
        # app = ode_examples.Lorenz(t_end=20)
        # app = ode_examples.Mathieu()
        # app = ode_examples.NonlinearMix()
        # app = ode_examples.Mathieu()
        sdata = newtondata.StoppingParamaters(maxiter=300, rtol=1e-6, bt_maxiter=50, bt_c=0.01, bt_omega=0.5, divx=1e20)
        solver = Newton_Ode(app, k=5, n0=600)
        x0 = solver.initial_guess()
        newton = newton.Newton(nd = solver, verbose=2, sdata=sdata, verbose_bt=True)
        xs, info = newton.solve(x0)
        if not info.success:
            print(f"@@@@@@@@@@@@@@@@ {info.failure}")
        solver.plot(xs)

