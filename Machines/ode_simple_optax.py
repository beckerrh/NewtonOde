import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax

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

# Points de collocation
t0, t1, n_colloc = 0, 3, 10
t_colloc = np.linspace(t0, t1, n_colloc)
# Machine
layers = [1, 8, 8, 1]
# Initialisation des paramètres
def init_params():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    params = []
    for l in range(1,len(layers)):
        in_dim, out_dim =  layers[l-1], layers[l]
        key, subkey = jax.random.split(key)
        W = jax.random.normal(subkey, (out_dim, in_dim)) * np.sqrt(2 / in_dim)
        b = np.zeros(out_dim)
        params.append((W,b))
    return params
def forward(params, t):
    t = np.array([t])
    for W, b in params[:-1]:
        t = np.tanh(W @ t + b)
    W, b = params[-1]
    return (W @ t +b)[0]
# Calcul de u''(x) par JAX autodiff
def dudt(params, t):
    return jax.grad(forward, argnums=1)(params, t)
def d2udt2(params, t):
    return jax.grad(dudt, argnums=1)(params, t)
def residual(params, t):
    lam = 0.9
    return dudt(params, t) - lam*forward (params, t)
    return d2udt2(params, t) + (np.pi ** 2) * np.sin(np.pi * t)
# Total loss = ode_colloc + boundary conditions
def loss(params):
    # Ode loss
    res = jax.vmap(lambda t: residual(params, t))(t_colloc)
    ode_loss = np.mean(res ** 2)
    # Boundary conditions
    bc_loss = (forward(params, t_colloc[0])-1) ** 2
    # bc_loss = forward(params, t_colloc[0]) ** 2 + forward(params, t_colloc[-1]) ** 2
    return ode_loss + bc_loss
def solution(t):
    lam = 0.9
    return np.exp(lam*t)

params = init_params()
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

n_epochs = 400
for epoch in range(n_epochs):
    params, opt_state, loss_value, grads, updates = train_step(params, opt_state)
    if epoch % 100== 0:
        print(f"Epoch {epoch:7d}, Loss: {loss_value:.3e}")

# Visu
t_plot = np.linspace(t0, t1, 200)
u_pred = jax.vmap(lambda t: forward(params,t))(t_plot)
u_true = solution(t_plot)

plot_solutions(t_plot, t_colloc, u_pred, u_true)
