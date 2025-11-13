import jax
import jax.numpy as jnp
import optax
from flax import nnx
import outils


#==================================================================
class MachineMlp(nnx.Module):
    def __init__(self, layers, key):
        """layers: list of hidden layer sizes, last entry = nbases"""
        super().__init__()
        keys = jax.random.split(key, len(layers) + 1)
        in_dim = 1
        self.layers = []
        for i, out_dim in enumerate(layers):
            if in_dim != out_dim:
                layer = nnx.Linear(in_dim, out_dim, rngs=nnx.Rngs(keys[i]))
            else:
                layer = nnx.Linear(in_dim, out_dim, rngs=nnx.Rngs(keys[i]))
                # layer = outils.TriangularDense(in_dim, rngs=nnx.Rngs(keys[i]))
                # layer = outils.BandedDense(in_dim, 3, rngs=nnx.Rngs(keys[i]))
            self.layers.append(layer)
            # self.layers.append(nnx.Dropout(out_dim, out_dim, rngs=nnx.Rngs(keys[i])))
            in_dim = out_dim
    def __call__(self, t):
        # t shape: scalar or (batch,)
        x = np.atleast_1d(t).reshape(-1, 1)  # shape (batch, 1)
        for layer in self.layers[:-1]:
            x = x + np.tanh(layer(x))
        last_layer = self.layers[-1]
        return last_layer(x)

    def regularization(self, t_colloc):
        M = self(t_colloc)  # shape (nbases, N)
        e = np.sum(M, axis=1) - 1
        return np.mean(e ** 2)

#==================================================================
class ModelEdo(nnx.Module):
    def __init__(self, app, nbases, key):
        self.app = app
        self.nbases = nbases
        self.nout = 1 if isinstance(app.x0, float) else len(app.x0)
        self.layer = nnx.Linear(nbases, self.nout, use_bias=False, rngs=nnx.Rngs(key))
    def __call__(self, t):
        return self.layer(t).T
    def forward(self, machine, t):
        return self(machine(t)).squeeze()
    def residual_edo_single(self, machine, t):
        u = self.forward(machine, t)
        dudt = jax.jacrev(self.forward, argnums=1)(machine, t)
        return dudt - self.app.f(u)
    def residual_edo(self, machine, t_colloc):
        return jax.vmap(lambda t: self.residual_edo_single(machine, t))(t_colloc)
    def residual_bdry(self, machine, t_colloc):
        return self.forward(machine, t_colloc[0])-self.app.x0

def train_all(machine, model, t_colloc, lr=0.01, n_epochs=1000):
    graphdef_machine, params_machine, batch_stats_machine = nnx.split(machine, nnx.Param, nnx.BatchStat)
    graphdef_model, params_model, batch_stats_model = nnx.split(model, nnx.Param, nnx.BatchStat)
    params = (params_machine, params_model)

    optimizer = optax.lbfgs(learning_rate=lr)
    opt_state = optimizer.init(params)

    def loss(params):
        params_machine, params_model = params
        # Create a *temporary* machine for this loss eval
        machine_tmp = nnx.merge(graphdef_machine, params_machine, batch_stats_machine)
        model_temp = nnx.merge(graphdef_model, params_model, batch_stats_model)
        res_dom = model_temp.residual_edo(machine_tmp, t_colloc)
        res_bdry = model_temp.residual_bdry(machine_tmp, t_colloc)
        return np.mean(res_dom**2) + np.mean(res_bdry**2) + 0.1*machine_tmp.regularization(t_colloc)

    @jax.jit
    def train_step(params, opt_state):
        loss_value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(
            grads, opt_state, params, value=loss_value, grad=grads, value_fn=loss
        )
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, grads, updates

    for epoch in range(n_epochs):
        params, opt_state, loss_value, grads, updates = train_step(params, opt_state)
        if epoch % 100 == 0:
            print(f"Epoch {epoch:7d}, Loss: {loss_value:.3e}")

    # Merge final trained parameters back into machine
    trained_machine = nnx.merge(graphdef_machine, params[0], batch_stats_machine)
    trained_model = nnx.merge(graphdef_model, params[1], batch_stats_model)
    return trained_machine, trained_model

if __name__ == '__main__':
    # Application
    import ode_examples
    # app, layers, n_colloc = ode_examples.Exponential(), [3,3], 6
    # app, layers, n_colloc = ode_examples.Exponential(), [11,11], 12
    # app, layers, n_colloc = ode_examples.Exponential(), [23,23], 24
    # app, layers, n_colloc = ode_examples.Pendulum(t_end=4), [24,24], 25
    # app, layers, n_colloc = ode_examples.Pendulum(t_end=5, is_linear=False), [24,24], 25
    app, layers, n_colloc = ode_examples.ExponentialJordan(), [8,8], 10
    # machine = Machine(layers)
    key = jax.random.PRNGKey(42)
    machine = MachineMlp(layers, key)
    key = jax.random.PRNGKey(43)
    model = ModelEdo(app, layers[-1], key)
    t_colloc = np.linspace(app.t_begin, app.t_end, n_colloc)

    machine, model = train_all(machine, model, t_colloc, 0.1, n_epochs=2000)

    # Plot results
    t_plot = np.linspace(t_colloc[0], t_colloc[-1], 200)
    u_pred = jax.vmap(lambda t: model.forward(machine, t))(t_plot)
    k1 = r"$u' = f(u)$ " + app.name
    if hasattr(app, 'solution'):
        u_exact = app.solution(t_plot)
        plot_dict = {k1: [t_plot, {"mach": u_pred, "exact":u_exact}]}
        err = u_exact-u_pred
        plot_dict["error"] = [t_plot, err]
    else:
        u_true = None
        import ode_solver
        cgp = ode_solver.CgK(k=2)
        u_node, u_coef = cgp.run_forward(t_colloc, app)
        u_cg = u_node.T
        u_machine = jax.vmap(lambda t: model.forward(machine, t))(t_colloc)
        plot_dict = {k1: [t_colloc, {"mach": u_machine, "cg":u_cg}]}
        err = u_cg-u_machine
        plot_dict["error"] = [t_colloc, err]
    print(f"err = {np.mean(err**2):.5e}")

    base_dict = machine(t_plot)
    plot_dict["bases"] = [t_plot, base_dict]
    outils.plot_solutions(plot_dict)