import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from ode_solver import ODE_Legendre
from Utility import mesh1d, plotting
import ode_examples
import matplotlib.pyplot as plt

#==================================================================
class OdeLinearized(ode_examples.OdeExample):
    def __init__(self, app):
        assert not hasattr(app, 'a_coef')
        assert not hasattr(app, 'b_coef')
        assert hasattr(app, 'f')
        assert hasattr(app, 'df')
        self.app = app
        super().__init__(u0=app.u0, t_begin=app.t_begin, t_end=app.t_end)



#==================================================================
class Newton_Ode():
    def __init__(self, app, k=0, mesh=None):
        self.solver = ODE_Legendre(k)
        self.app = OdeLinearized(app)
        if not mesh:
            self.mesh = mesh1d.mesh(app.t_begin, app.t_end, n=4)
        else:
            self.mesh = mesh
    def initial_guess(self):
        nt = self.mesh.shape[0]
        nbasis = len(self.solver.phi)
        ncomp = 1 if np.ndim(self.app.u0) == 0 else len(self.app.u0)
        return np.zeros(shape=(nt, nbasis, ncomp))


#==================================================================
def test_interpolation(niter=6):
    app = ode_examples.PolynomialIntegration(degree=9, ncomp=1)
    solver = ODE_Legendre(k=3)
    mesh = mesh1d.mesh(app.t_begin, app.t_end, n=4)
    for iter in range(niter):
        ucoeff = solver.run(mesh, app.u0, app.a_coef, app.b_coef)
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
    mesh = mesh1d.mesh(app.t_begin, app.t_end, n=4)
    ns, errs_l2, errs_disc, etas = [], [], [], []
    for iter in range(niter):
        ucoeff = solver.run_linear(mesh, app)
        el2, edisc, err_cell = solver.compute_error(mesh, ucoeff, app.solution)
        eta, eta_cell = solver.estimator(mesh, ucoeff, app)
        errs_l2.append(el2)
        etas.append(eta)
        # print(f"{el2/eta=}")
        ns.append(mesh.shape[0])
        if plot:
            t_mp, u_mp = solver.interpolate_midpoint(mesh, ucoeff)
            u_true = app.solution(t_mp).squeeze()
            pd = {app.name: {}, "Mesh": {}, "Estimator": {}}
            pd[app.name]['x'] = t_mp
            pd[app.name]['y'] = {'app': u_mp, 'sol': u_true}
            pd[app.name]['kwargs'] = {'app': {'marker': 'o'}}
            pd["Mesh"]['x'] = mesh
            pd["Mesh"]['y'] = {'1/h': 1/(mesh[1:]-mesh[:-1])}
            pd["Mesh"]['type'] = 'step'
            pd["Estimator"]['x'] = t_mp
            pd["Estimator"]['y'] = {'eta': eta_cell, 'err': err_cell}
            plotting.plot_solutions(pd, title=f"Iteration {iter} N={mesh.shape[0]}")
            plt.show()
        mesh_new, refinfo = mesh1d.adapt_mesh(mesh, eta_cell)
        ucoeff_new = solver.interpolate(ucoeff, mesh_new, refinfo)
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

    if todo == 'interpolation':
        test_interpolation()
    elif todo == 'linear':
        solver = ODE_Legendre(k=2)
        # app = ode_examples.PolynomialIntegration(degree=8, ncomp=3)
        # app = ode_examples.Exponential(lam=10.2)
        # app = ode_examples.ExponentialJordan(lam=2.2)
        # app = ode_examples.TimeDependentRotation()
        app = ode_examples.RotScaleForce()
        test_adaptive_linear(app, solver, niter=12)
    else:
        from Newton import newton
        app = ode_examples.Logistic()
        solver = Newton_Ode(app, k=2)
        x0 = solver.initial_guess()
        newton = newton.Newton(nd = solver)
        newton.solve(x0)
