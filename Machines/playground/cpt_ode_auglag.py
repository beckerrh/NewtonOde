# compare_full_vs_triangular.py
# (Script identical to the notebook cell attempted in the environment)
import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# -------------------- Config --------------------
key = jax.random.PRNGKey(0)
u0 = 1.5
t_domain = (0.0, 1.0)
hidden = 64
lr = 1e-3
mu0 = 5.0
n_steps = 2000
batch_size = 64
print_every = 200

# -------------------- Utilities --------------------
def time_to_input(ts):
    ts = np.asarray(ts)
    if ts.ndim == 1:
        ts = ts.reshape(-1,1)
    return ts

def exact_solution(ts):
    return u0 * np.exp(-ts)

# -------------------- Triangular Dense module --------------------
class TriangularDense(nnx.Module):
    def __init__(self, in_dim, out_dim, key, lower=True, use_bias=True):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lower = lower
        self.use_bias = use_bias
        k1, = jax.random.split(key, 1)
        w_init = jax.random.normal(k1, (out_dim, in_dim)) * np.sqrt(2.0 / max(1, in_dim))
        self.weight = nnx.Param(w_init)
        if use_bias:
            self.bias = nnx.Param(np.zeros((out_dim,)))
        else:
            self.bias = None
        if lower:
            mask = np.tril(np.ones((out_dim, in_dim)))
        else:
            mask = np.triu(np.ones((out_dim, in_dim)))
        self.mask = mask

    def __call__(self, x):
        W = self.weight * self.mask
        y = x @ W.T
        if self.use_bias:
            y = y + self.bias
        return y

# -------------------- Simple Dense --------------------
class SimpleDense(nnx.Module):
    def __init__(self, in_dim, out_dim, key, use_bias=True):
        k1, = jax.random.split(key, 1)
        w_init = jax.random.normal(k1, (out_dim, in_dim)) * np.sqrt(2.0 / max(1, in_dim))
        self.weight = nnx.Param(w_init)
        if use_bias:
            self.bias = nnx.Param(np.zeros((out_dim,)))
        else:
            self.bias = None

    def __call__(self, x):
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y

# -------------------- Two MLP constructors --------------------
def make_full_mlp(key, hidden=64):
    k1, k2, k3 = jax.random.split(key, 3)
    class MLP_full(nnx.Module):
        def __init__(self):
            self.l1 = SimpleDense(1, hidden, k1)
            self.l2 = SimpleDense(hidden, hidden, k2)
            self.l3 = SimpleDense(hidden, 1, k3)
        def __call__(self, t):
            x = jax.nn.relu(self.l1(t))
            x = jax.nn.relu(self.l2(x))
            return self.l3(x)
    return MLP_full()

def make_triangular_mlp(key, hidden=64):
    k1, k2, k3 = jax.random.split(key, 3)
    class MLP_tri(nnx.Module):
        def __init__(self):
            self.l1 = SimpleDense(1, hidden, k1)
            self.l2 = TriangularDense(hidden, hidden, k2, lower=True)
            self.l3 = SimpleDense(hidden, 1, k3)
        def __call__(self, t):
            x = jax.nn.relu(self.l1(t))
            x = jax.nn.relu(self.l2(x))
            return self.l3(x)
    return MLP_tri()

# -------------------- Init models and optimizers --------------------
key1, key2 = jax.random.split(key)
model_full = make_full_mlp(key1, hidden=hidden)
model_tri  = make_triangular_mlp(key2, hidden=hidden)

opt_full = optax.adam(lr)
opt_tri  = optax.adam(lr)
opt_state_full = opt_full.init(model_full.parameters())
opt_state_tri  = opt_tri.init(model_tri.parameters())

# -------------------- ODE utilities --------------------
def dudt_scalar_batch(model, ts):
    def single_dudt(t_single):
        return jax.grad(lambda s: model(s.reshape(1,1)).squeeze())(t_single)
    return jax.vmap(single_dudt)(ts).squeeze(-1)

def residual_loss(model, ts):
    ts = time_to_input(ts)
    us = model(ts).squeeze(-1)
    dudt = dudt_scalar_batch(model, ts)
    resid = dudt + us
    return np.mean(resid**2)

def constraint_value(model):
    t0 = np.array([[0.0]])
    return model(t0).squeeze()

def augmented_loss_params(params_like, model_obj, lam, mu, ts):
    mtmp = model_obj.update_parameters(params_like)
    res = residual_loss(mtmp, ts)
    h = constraint_value(mtmp) - u0
    aug = lam * h + 0.5 * mu * (h**2)
    return res + aug, (res, h)

@jax.jit
def train_step(model, opt_state, optimizer, lam, mu, ts):
    params = model.parameters()
    (loss_val, (res_val, h_val)), grads = jax.value_and_grad(augmented_loss_params, has_aux=True)(
        params, model, lam, mu, ts)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    model = model.update_parameters(new_params)
    lam = lam + mu * h_val
    return model, opt_state, lam, loss_val, res_val, h_val

# -------------------- Training --------------------
rng = np.random.default_rng(1234)
mu_full = mu0; mu_tri = mu0
lam_full = 0.0; lam_tri = 0.0

history = {"step": [], "loss_full": [], "res_full": [], "h_full": [],
                     "loss_tri": [], "res_tri": [], "h_tri": [], "mu_full": [], "mu_tri": []}

for step in range(1, n_steps+1):
    ts_batch = rng.uniform(t_domain[0], t_domain[1], size=(batch_size,1)).astype(np.float32)
    model_full, opt_state_full, lam_full, loss_f, res_f, h_f = train_step(
        model_full, opt_state_full, opt_full, lam_full, mu_full, ts_batch)
    model_tri, opt_state_tri, lam_tri, loss_t, res_t, h_t = train_step(
        model_tri, opt_state_tri, opt_tri, lam_tri, mu_tri, ts_batch)

    if step % print_every == 0 or step == 1:
        print(f"step {step:4d} | full loss {float(loss_f):.3e} res {float(res_f):.3e} h {float(h_f):.3e} | tri loss {float(loss_t):.3e} res {float(res_t):.3e} h {float(h_t):.3e}")

    history["step"].append(step)
    history["loss_full"].append(float(loss_f)); history["res_full"].append(float(res_f)); history["h_full"].append(float(h_f))
    history["loss_tri"].append(float(loss_t)); history["res_tri"].append(float(res_t)); history["h_tri"].append(float(h_t))
    history["mu_full"].append(mu_full); history["mu_tri"].append(mu_tri)

    if step % 600 == 0:
        hnorm_f = abs(float(h_f)); hnorm_t = abs(float(h_t))
        if hnorm_f > 1e-4:
            mu_full *= 2.0
        if hnorm_t > 1e-4:
            mu_tri *= 2.0

# -------------------- Evaluate and plot --------------------
ts_plot = np.linspace(0,1,200).reshape(-1,1).astype(np.float32)
us_exact = np.array(exact_solution(ts_plot))

us_full = np.array(model_full(time_to_input(ts_plot))).squeeze(-1)
us_tri  = np.array(model_tri(time_to_input(ts_plot))).squeeze(-1)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.plot(ts_plot.squeeze(-1), us_exact.squeeze(-1), label="Exact", linestyle='--')
plt.plot(ts_plot.squeeze(-1), us_full, label="Full MLP")
plt.plot(ts_plot.squeeze(-1), us_tri, label="Triangular MLP")
plt.xlabel("t"); plt.ylabel("u(t)")
plt.title("Learned solution vs exact (u'=-u, u(0)=u0)")
plt.legend(); plt.grid(True); plt.show()

steps = np.array(history["step"])
plt.figure(figsize=(8,4))
plt.semilogy(steps, np.abs(np.array(history["h_full"])), label="|h| full")
plt.semilogy(steps, np.abs(np.array(history["h_tri"])), label="|h| tri")
plt.xlabel("training step"); plt.ylabel("|h| = |u(0)-u0|")
plt.title("Constraint violation during training (semilog)")
plt.legend(); plt.grid(True); plt.show()

print("Final u(0) full  :", float(constraint_value(model_full)))
print("Final u(0) tri   :", float(constraint_value(model_tri)))
print("Target u0        :", float(u0))
err_full = np.linalg.norm(us_full - us_exact.squeeze(-1)) / np.sqrt(len(us_full))
err_tri  = np.linalg.norm(us_tri  - us_exact.squeeze(-1)) / np.sqrt(len(us_tri))
print(f"Final L2 error (full) {err_full:.3e}, (tri) {err_tri:.3e}")