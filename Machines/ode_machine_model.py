import jax.numpy as jnp
import jax
import optax
from flax import nnx
import plotting

#-----------------------------------------------------------
class MLP(nnx.Module):
    def __init__(self, layers, key):
        key, subkey = jax.random.split(key)
        self.layers = []
        for l in range(1,len(layers)):
            in_dim, out_dim =  layers[l-1], layers[l]
            key, subkey = jax.random.split(key)
            layer = nnx.Linear(in_dim, out_dim, rngs=nnx.Rngs(key))
            self.layers.append(layer)
    def normalize_t(self, t):
        t_mean, t_std = jnp.mean(t_colloc), jnp.std(t_colloc)
        return (t - t_mean) / t_std
    def basis(self, t):
        t = jnp.atleast_1d(t).reshape(-1, 1)
        t = self.normalize_t(t)
        for layer in self.layers[:-1]:
            t = jnp.tanh(layer(t))
        return t.T.squeeze()
    def forward(self, t):
        t = self.basis(t)
        last_layer = self.layers[-1]
        return last_layer(t).squeeze()
    def regularization(self, t_colloc):
        M = self.basis(t_colloc)  # shape (nbases, N)
        # print(f"{M.shape=}")
        e = jnp.sum(M, axis=1) - 1
        return jnp.mean(e ** 2)

#-----------------------------------------------------------
class ModelOde:
    def __init__(self, t_colloc):
        self.lam = 0.4
        self.t_colloc, self.t0 = t_colloc, t_colloc[0]
        self.A = jnp.array([[self.lam, 1.0], [0.0, self.lam]])
    def residual_ode_single(self, machine, t):
        u = machine.forward(t)
        dudt = jax.jacrev(machine.forward)(t)
        return dudt - self.A@u
    def residual_ode(self, machine):
        return jax.vmap(lambda t: self.residual_ode_single(machine, t))(self.t_colloc)
    def residual_bc(self, machine):
        return machine.forward(self.t0)-1
    def solution(self, t):
        exp = jnp.exp(self.lam*t)
        return jnp.array([exp*(1+t), exp]).T


# Points de collocation
t0, t1, n_colloc = 0, 3, 10
t_colloc = jnp.linspace(t0, t1, n_colloc)
# Machine
layers = [1, 6, 6, 2]
key = jax.random.PRNGKey(33)
machine = MLP(layers, key)
graphdef, params, batch_stats = nnx.split(machine, nnx.Param, nnx.BatchStat)
model = ModelOde(t_colloc)

def loss(params):
    # ode loss
    machine_tmp = nnx.merge(graphdef, params, batch_stats)
    ode_res = model.residual_ode(machine_tmp)
    bc_loss = model.residual_bc(machine_tmp)
    return 100*jnp.mean(ode_res**2) + jnp.mean(bc_loss**2) + machine_tmp.regularization(model.t_colloc)




optimizer = optax.lbfgs(learning_rate=0.001)
opt_state = optimizer.init(params)
@jax.jit
def train_step(params, opt_state):
    loss_value, grads = jax.value_and_grad(loss)(params)
    updates, opt_state = optimizer.update(
        grads, opt_state, params, value=loss_value, grad=grads, value_fn=loss
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value, grads, updates

n_epochs = 1200
for epoch in range(n_epochs):
    params, opt_state, loss_value, grads, updates = train_step(params, opt_state)
    if epoch % 100== 0:
        print(f"Epoch {epoch:7d}, Loss: {loss_value:.3e}")

trained_machine = nnx.merge(graphdef, params, batch_stats)
# Visu
t_plot = jnp.linspace(t0, t1, 200)
u_pred = jax.vmap(trained_machine.forward)(t_plot)
u_true = model.solution(t_plot)

plot_dict = {"u* vs. uh" : {'t_plot':t_plot, 'u_plot':{'u*':u_true, 'uh':u_pred}}}
plot_dict['e'] = {'t_plot':t_plot, 'u_plot':{'e': u_true-u_pred}}
plotting.plot_solutions(plot_dict)
