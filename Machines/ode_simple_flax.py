import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx

# Commençons par la fonction de visu
def plot_solutions(t_plot, t_colloc, u1, u2, t1='Approximation', t2='Solution', ls1='', ls2='--'):
    plt.plot(t_plot, u1, ls1, label=t1)
    plt.plot(t_plot, u2, ls2, label=t2)
    plt.plot(t_colloc, np.zeros_like(t_colloc), 'Xr', label='t_colloc')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title(r"$u'' = -\pi^2 \sin(\pi t)$")
    plt.grid(True)
    plt.show()

#-----------------------------------------------------------
class MLP(nnx.Module):
    def __init__(self, layers, key):
        self.lam = 0.9
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
    def forward(self, t):
        t = jnp.array([t])
        t = self.normalize_t(t)
        for layer in self.layers[:-1]:
            t = jnp.tanh(layer(t))
        last_layer = self.layers[-1]
        return last_layer(t)[0]
    def dudt(self, t):
        return jax.grad(self.forward)(t)
    def residual_ode_colloc(self, t):
        return self.dudt(t) - self.lam*self.forward(t)
    def residual_bc(self, t0):
        return (self.forward(t0)-1) **2
    def solution(self, t):
        return jnp.exp(self.lam*t)


# Points de collocation
t0, t1, n_colloc = 0, 3, 10
t_colloc = jnp.linspace(t0, t1, n_colloc)
# Machine
layers = [1, 8, 8, 1]
key = jax.random.PRNGKey(33)
machine = MLP(layers, key)
graphdef, params, batch_stats = nnx.split(machine, nnx.Param, nnx.BatchStat)

def loss(params):
    # ode loss
    machine_tmp = nnx.merge(graphdef, params, batch_stats)
    res = jax.vmap(machine_tmp.residual_ode_colloc)(t_colloc)
    ode_loss = jnp.mean(res ** 2)
    # return ode_loss
    # Boundary conditions
    bc_loss = machine_tmp.residual_bc(t_colloc[0])
    return 100*ode_loss + bc_loss




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
u_true = trained_machine.solution(t_plot)

plot_solutions(t_plot, t_colloc, u_pred, u_true)
