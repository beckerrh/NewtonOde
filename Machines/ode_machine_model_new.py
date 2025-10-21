import jax.numpy as jnp
import jax
import optax
from flax import nnx
import training

#-----------------------------------------------------------
class MLP(nnx.Module):
    t_mean: float = nnx.field(init=False, static=True)
    t_std: float = nnx.field(init=False, static=True)
    def __init__(self, layers, key, t_colloc):
        self.t_mean, self.t_std = float(jnp.mean(t_colloc)), float(jnp.std(t_colloc))
        key, subkey = jax.random.split(key)
        self.feature_layers = []
        for l in range(1, len(layers) - 1):  # all but last
            in_dim, out_dim = layers[l - 1], layers[l]
            key, subkey = jax.random.split(key)
            self.feature_layers.append(
                nnx.Linear(in_dim, out_dim, rngs=nnx.Rngs(key))
            )
        in_dim, out_dim = layers[-2], layers[-1]
        self.final_layer = nnx.Linear(in_dim, out_dim, rngs=nnx.Rngs(subkey))
    def normalize_t(self, t):
        return (t - self.t_mean) / self.t_std
    def basis(self, t):
        """Compute nonlinear basis functions φ(t)."""
        t = jnp.atleast_1d(t).reshape(-1, 1)
        t = self.normalize_t(t)
        for layer in self.feature_layers:
            t = t+jnp.tanh(layer(t))
        return t.T.squeeze()
    def basis_t(self, t):
        dphi_dt_single = lambda ti: jax.jacrev(self.basis)(ti).squeeze()
        return jax.vmap(dphi_dt_single)(t).T
    def forward(self, t):
        """Full forward pass: linear combination of basis."""
        phi_t = self.basis(t)
        return self.final_layer(phi_t).squeeze()
    def regularization_basis(self, t_colloc, level=0):
        M = self.basis(t_colloc)
        print(f"{M.shape=}")
        r1 = jnp.sum(M, axis=0)
        if level==0: return  jnp.mean(r1 ** 2)
        r2 = M.T@M - jnp.eye(M.shape[1])
        if level==1: return  jnp.mean(r1 ** 2) + 0.01*jnp.mean(r2 ** 2)
        N = self.basis_t(t_colloc)
        r3 = N@N.T - jnp.eye(N.shape[0])
        return  jnp.mean(r1 ** 2) + 0.01*jnp.mean(r2 ** 2) + 0.0001*jnp.mean(r3**2)
    def forward_batch(self, t_colloc):
        return jax.vmap(self.forward)(t_colloc)



#-----------------------------------------------------------
class ModelOde:
    def __init__(self, app, n_colloc):
        self.app = app
        self.t0, self.t1 = app.t_begin, app.t_end
        self.t_colloc = jnp.linspace(self.t0, self.t1, n_colloc)
    def residual_ode_single(self, machine, t):
        u = machine.forward(t)
        dudt = jax.jacrev(machine.forward)(t)
        return dudt - self.app.f(u)
    def residual_ode(self, machine):
        return jax.vmap(lambda t: self.residual_ode_single(machine, t))(self.t_colloc)
    def residual_bc(self, machine):
        return machine.forward(self.t0)-self.app.u0

#-----------------------------------------------------------
def solve_ode(app, layers, n_colloc, return_basis=False):
    key = jax.random.PRNGKey(34)
    model = ModelOde(app, n_colloc)
    machine = MLP(layers, key, model.t_colloc)
    if return_basis: base0 = machine.basis(model.t_colloc).T
    def ode_loss(machine_tmp):
        ode_res = jnp.mean(model.residual_ode(machine_tmp))**2
        bc_loss = jnp.mean(model.residual_bc(machine_tmp))**2
        return ode_res + bc_loss+ 0.5 * machine_tmp.regularization_basis(model.t_colloc,level=0)
    trained_machine = training.train_machine_and_solve(machine, ode_loss)
    if return_basis:
        base1 = trained_machine.basis(model.t_colloc).T
        return trained_machine, model, base0, base1
    return trained_machine

if __name__ == '__main__':
    import plotting, ode_examples
    # app, layers, n_colloc = ode_examples.Exponential(), [3,3], 6
    # app, layers, n_colloc = ode_examples.Exponential(), [11,11], 12
    # app, layers, n_colloc = ode_examples.Exponential(), [23,23], 24
    # app, layers, n_colloc = ode_examples.Pendulum(t_end=4), [24,24], 25
    # app, layers, n_colloc = ode_examples.Pendulum(t_end=5, is_linear=False), [24,24], 25
    app, layers, n_colloc = ode_examples.Logistic(), [8,8], 10
    # app, layers, n_colloc = ode_examples.ExponentialJordan(t_end=10, lam=-0.2), [32,32], 100

    layers = [1,*layers, app.ncomp]

    allinone = False
    if allinone:
        trained_machine, model, base0, base1 = solve_ode(app, layers, n_colloc, return_basis=True)
    else:
        machine = MLP(layers, jax.random.PRNGKey(0), t_colloc=jnp.linspace(0, 1, 10))
        model = ModelOde(app, n_colloc)
        def machine_loss(machine_tmp):
            return machine_tmp.regularization_basis(model.t_colloc,level=1)
        machine = training.train_features(machine, machine_loss)
        base0 = machine.basis(model.t_colloc).T
        def ode_loss(machine_tmp):
            ode_res = jnp.mean(model.residual_ode(machine_tmp)) ** 2
            bc_loss = jnp.mean(model.residual_bc(machine_tmp)) ** 2
            return ode_res + bc_loss
        trained_machine = training.solve_coefficients(machine, ode_loss)
        base1 = trained_machine.basis(model.t_colloc).T

    t_plot = jnp.linspace(app.t_begin, app.t_end, 4 * n_colloc)
    u_mlp = trained_machine.forward_batch(t_plot)

    plot_dict = {"u* vs. uh" : {'t_plot':t_plot, 'u_plot':{}}}
    plot_dict["u* vs. uh"]['u_plot']['u_mlp'] = u_mlp

    if hasattr(app, 'solution'):
        u_true = app.solution(t_plot)
        plot_dict["u* vs. uh"]['u_plot']['u*'] = u_true
        plot_dict['e'] = {'t_plot': t_plot, 'u_plot': {}}
        plot_dict['e']['u_plot']['e'] =  u_mlp - u_true
        plot_dict['e']['u_plot']['abs(e)'] =  jnp.abs(u_mlp - u_true)

    plot_dict["bases_0"] = {'t_plot': model.t_colloc, 'u_plot': {'b': base0}, 'no_legend': True}
    plot_dict["bases"] = {'t_plot': model.t_colloc, 'u_plot': {}, 'no_legend': True}
    plot_dict["bases"]['u_plot']['b'] = base1


    plotting.plot_solutions(plot_dict)
