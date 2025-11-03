import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


#==================================================================
def plot_solutions(t_plot, u_pred, u_true, t_colloc):
    plt.plot(t_plot, u_pred, label='Approximation')
    plt.plot(t_plot, u_true, '--', label='Solution')
    plt.plot(t_colloc, np.zeros_like(t_colloc), 'Xr', label='t_colloc')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title(r"$u'' = -\pi^2 \sin(\pi t)$")
    plt.grid(True)
    plt.show()

#==================================================================
def train(params, update, loss, n_epoch=1000, lr=0.001, rtol=1e-6, gtol=1e-9, out=30, dtol=1e-12):
    # torch.autograd.set_detect_anomaly(True)
    trn_loss_history = []
    if out > n_epoch:
        filter = 1
    elif out == 0:
        filter = n_epoch
    else:
        filter = n_epoch // out
    print(f"{'Iter':^6s}  {'loss':^12s} {'diff':^12s}")
    for epoch in range(n_epoch):
        params = update(params, lr=lr)
        diffres = np.nan
        if epoch == 0:
            loss_val = loss(params)
            tol = max(rtol * loss_val, gtol)
            trn_loss_history.append(loss_val)
        if out and epoch % filter == 0:
            loss_val = loss(params)
            diffres = abs(loss_val - trn_loss_history[-1])
            print(f"{epoch:6d} {loss_val:12.3e} {diffres:12.3e}")
            trn_loss_history.append(loss_val)
        if loss_val < tol:
            return trn_loss_history
        if epoch and loss_val < tol and diffres < dtol:
            return trn_loss_history
    return params, trn_loss_history

#==================================================================
class MachineEdoO2:
    def _layer(self, in_dim, out_dim, key, bzero=False):
        key, subkey = jax.random.split(key)
        W = jax.random.normal(subkey, (out_dim, in_dim)) * np.sqrt(2 / in_dim)
        if bzero: return key, W
        b = np.zeros(out_dim)
        return key, (W, b)
    def __init__(self, layers, t_colloc):
        self.t_colloc = t_colloc
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        params = []
        for l in range(1,len(layers)):
            in_dim, out_dim =  layers[l-1], layers[l]
            key, param = self._layer(in_dim, out_dim, key)
            params.append(param)
        self.params = params
    def forward(self, params, t):
        t = np.array([t])
        for W, b in params[:-1]:
            t = np.tanh(W @ t + b)
        W, b = params[-1]
        return (W @ t + b)[0]
    def predict(self, t):
        return self.forward(self.params, t)
    # Compute u''(x) using JAX autodiff
    def dudt(self, params, t):
        return jax.grad(self.forward, argnums=1)(params, t)
    def d2udt2(self, params, t):
        return jax.grad(self.dudt, argnums=1)(params, t)
    def residual(self, params, t):
        return self.d2udt2(params, t) + (np.pi ** 2) * np.sin(np.pi * t)
    # Total loss = physics + boundary conditions
    def loss(self, params):
        # Physics loss
        res = jax.vmap(lambda t: self.residual(params, t))(self.t_colloc)
        physics_loss = np.mean(res ** 2)
        # Boundary conditions: u(0)=0, u(1)=0
        bc_loss = m.forward(params, self.t_colloc[0]) ** 2 + m.forward(params, self.t_colloc[-1]) ** 2
        return physics_loss + bc_loss


# Machine setup
layers = [1, 8, 8, 1]
# Collocation points
n_colloc = 10
t_colloc = np.linspace(0, 3, n_colloc)
m = MachineEdoO2(layers, t_colloc)
# Training step
@jax.jit
def update(params, lr):
    grads = jax.grad(m.loss)(params)
    return  [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]
#Training loop
#(m.params, m.coeff), trn_loss_history = train((m.params,m.coeff), update, loss=m.loss, n_epoch=6000)
for epoch in range(6000):
    m.params = update(m.params, lr=0.001)
    if epoch % 100 == 0:
        loss_val = m.loss(m.params)
        print(f"{epoch:6d} {loss_val:12.3e}")

# Plot results
t_plot = np.linspace(t_colloc[0], t_colloc[-1], 200)
u_pred = jax.vmap(lambda t: m.predict(t))(t_plot)
u_true = np.sin(np.pi * t_plot)

plot_solutions(t_plot, u_pred, u_true, t_colloc)