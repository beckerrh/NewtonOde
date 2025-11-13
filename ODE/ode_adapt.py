import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from ode_solver import ODE_Legendre
from Utility import mesh1d, plotting
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
    pd[app.name]['kwargs'] = {'app': {'marker': 'o'}}
    pd["Mesh"]['x'] = mesh
    pd["Mesh"]['y'] = {'1/h': 1.0 / (mesh[1:] - mesh[:-1])}
    pd["Mesh"]['type'] = 'step'
    cand_cell = ['zeta', 'eta', 'err']
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
    plt.show()


#==================================================================
class Newton_Ode():
    def __init__(self, app, k=0, mesh=None, n0=12):
        self.solver = ODE_Legendre(k)
        self.app = app
        if not mesh:
            self.mesh = mesh1d.mesh(app.t_begin, app.t_end, n=n0)
        else:
            self.mesh = mesh
    def initial_guess(self):
        nt = self.mesh.shape[0]
        nbasis = len(self.solver.phi)
        ncomp = 1 if np.ndim(self.app.x0) == 0 else len(self.app.x0)
        x0 = np.asarray(self.app.x0)
        x0all = np.zeros(shape=(nt-1, nbasis, ncomp))
        x0all[:,0] = x0
        # print(f"{x0=} {x0all=}")
        return (x0all, x0)
    def add_update(self, x, step, p):
        xnew = (x[0]+step*p[0], x[1]+step*p[1])
        return xnew
    def computeResidual(self, x):
        res = self.solver.compute_residual(self.mesh, self.app, x)
        # print(f"computeResidual {x=}\n{res=}")
        resnorm = np.linalg.norm(res[0])+np.linalg.norm(res[1])
        return SimpleNamespace(residual=res,
                               meritvalue=resnorm,
                               residual_norm=resnorm,
                               x_norm=self.solver.x_norm(self.mesh, x))
    def computeUpdate(self, r, x, info):
        p = self.solver.solve_linearized(self.mesh, self.app, x, r)
        # print(f"computeUpdate {x=}\n{p=}")
        return SimpleNamespace(update=p,
                               update_norm=self.solver.x_norm(self.mesh, p),
                               meritgrad=-np.sum(r[0]**2))
    def call_back(self, **kwargs):
        x = kwargs["x"]
        iterdata = kwargs["iterdata"]
        plot_all(mesh=self.mesh, app=self.app, x=x, solver=self.solver,
                 title=f"{self.app.name} Iter {iterdata.iter}")

    def call_back_baxktrack_failed(self, **kwargs):
        btresult = kwargs['btresult']
        dx = kwargs['dx']
        r = kwargs['r']
        if not btresult.success:
            print(f"{btresult.step=}")
        else:
            print(f"{kwargs['dxnorm']=} {kwargs['dxnorm_old']=}")




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
def test_adaptive_linear(app, solver, niter=6, plot=True):
    mesh = mesh1d.mesh(app.t_begin, app.t_end, n=3)
    ns, errs_l2, errs_disc, etas = [], [], [], []
    for iter in range(niter):
        x = solver.solve_linear(mesh, app)
        # el2, edisc, err_cell = solver.compute_error(mesh, x, app.solution)
        p = (np.zeros_like(x[0]), np.zeros_like(x[1]))
        zeta, zeta_cell, eta, eta_cell = solver.estimator(mesh, x, p, app)
        el2, err_cell = solver.compute_error(mesh, x, app.solution)
        errs_l2.append(el2)
        etas.append(eta)
        # print(f"{el2/eta=}")
        ns.append(mesh.shape[0])
        if plot:
            plot_all(mesh=mesh, app=app, solver=solver,
                     x=x, zeta=zeta_cell, eta=eta_cell, err=err_cell, title=f"Iteration {iter} N={mesh.shape[0]}")
        mesh_new, refinfo = mesh1d.adapt_mesh(mesh, eta_cell)
        x_new = solver.interpolate(x, mesh_new, refinfo)
        mesh = mesh_new
    ns = np.array(ns)
    errs_l2 = np.array(errs_l2)
    etas = np.array(etas)
    pdict = {'x': ns, 'y': {'L2': errs_l2, 'eta': etas}}
    plot_dict = {"Errors":  pdict}
    plotting.plot_error_curves(plot_dict)
    plt.show()

#==================================================================
def test_adaptive(app, solver, niter=6, plot=True):
    pass

#------------------------------------------------------------------
if __name__ == "__main__":
    todo = 'linear'
    np.random.seed(12)
    if todo == 'interpolation':
        test_interpolation()
    elif todo == 'nonlinear':
        solver = ODE_Legendre(k=0)
        # app = ode_examples.LinearIntegration()
        # app = ode_examples.PolynomialIntegration(degree=8, ncomp=2)
        # app = ode_examples.Exponential(lam=10.2)
        app = ode_examples.ExponentialJordan(lam=2.2)
        # app = ode_examples.TimeDependentRotation()
        # app = ode_examples.RotScaleForce()
        test_adaptive_linear(app, solver, niter=4)
    else:
        from Newton import newton, newtondata
        # app = ode_examples.PolynomialIntegration(degree=8, ncomp=2)
        # app = ode_examples.Exponential(lam=0.2)
        # app = ode_examples.ExponentialJordan(lam=2.2)
        # app = ode_examples.Logistic()
        # app = ode_examples.Pendulum(t_end=2)
        # app = ode_examples.VanDerPol(t_end=8)
        app = ode_examples.Robertson()
        # app = ode_examples.Lorenz()
        # app = ode_examples.Mathieu()
        # app = ode_examples.NonlinearMix()
        # app = ode_examples.Mathieu()
        sdata = newtondata.StoppingParamaters(bt_maxiter=50, bt_c=0.01)
        solver = Newton_Ode(app, k=1, n0=20)
        x0 = solver.initial_guess()
        newton = newton.Newton(nd = solver, verbose=2, sdata=sdata)
        newton.solve(x0)
