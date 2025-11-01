import jax.numpy as jnp
import ode_machine_model
from Utility import plotting
import ODE.ode_examples as ode_examples
import ode_cg

# from ODE import ode_examples
import scipy.integrate
import time

assert __name__ == '__main__'

# Application
# app, layers, n_colloc = ode_examples.Exponential(), [3,3], 6
# app, layers, n_colloc = ode_examples.Exponential(), [11,11], 12
# app, layers, n_colloc = ode_examples.Exponential(), [23,23], 24
# app, layers, n_colloc = ode_examples.Pendulum(t_end=4), [24,24], 25
# app, layers, n_colloc = ode_examples.Pendulum(t_end=5, is_linear=False), [24,24], 25
# app, layers, n_colloc = ode_examples.Logistic(), [8,8], 10
app, layers, n_colloc = ode_examples.ExponentialJordan(lam=-0.1, t_end=20), [64, 64], 200

layers = [1,*layers, app.ncomp]
t_plot = jnp.linspace(app.t_begin, app.t_end, 4*n_colloc)
t0 = time.time()

trained_machine = ode_machine_model.solve_ode(app, layers, n_colloc)
u_mlp = trained_machine.forward_batch(t_plot)
t1 = time.time()

cgp = ode_cg.CgK(k=2)
u_node, u_coef = cgp.run_forward(t_plot, app)
u_cg = u_node.T.squeeze()
t2 = time.time()

def f(y,t): return app.f(y)
u_odeint = scipy.integrate.odeint(f, app.u0, t_plot)
u_odeint = u_odeint.squeeze(-1) if u_odeint.shape[-1] == 1 else u_odeint
t3 = time.time()

print(f"Timing: mlp = {round(t1-t0,3)}s, cg = {round(t2-t1,3)}s, odeint = {round(t3-t2,3)}s")

print(f"{u_mlp.shape=} {u_cg.shape=} {u_odeint.shape=}")

plot_dict = {"u* vs. uh" : {'x':t_plot, 'y':{}}}
plot_dict["u* vs. uh"]['y']['u_mlp'] = u_mlp
plot_dict["u* vs. uh"]['y']['u_cg'] = u_cg
plot_dict["u* vs. uh"]['y']['u_odeint'] = u_odeint

if hasattr(app, 'solution'):
    u_true = app.solution(t_plot)
    plot_dict["u* vs. uh"]['y']['u*'] = u_true
    plot_dict['e'] = {'x': t_plot, 'y': {}}
    plot_dict['e']['y']['cg'] = u_cg - u_true
    plot_dict['e']['y']['mlp'] =  u_mlp - u_true
    plot_dict['e']['y']['odeint'] =  u_odeint - u_true

plotting.plot_solutions(plot_dict)
