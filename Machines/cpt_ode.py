# Full example: system ODE + AL + mu scheduling + collocation batching + optional lambda optimizer
import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx
from functools import partial
import numpy as np

# -----------------------
# Model: vector-valued MLP (maps t -> R^d)
# -----------------------
class MLP(nnx.Module):
    def __init__(self, key, in_dim=1, hidden=128, out_dim=3):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.l1 = nnx.Linear(in_dim, hidden, key=k1)
        self.l2 = nnx.Linear(hidden, hidden, key=k2)
        self.l3 = nnx.Linear(hidden, out_dim, key=k3)

    def __call__(self, t):
        # t expected shape (batch, 1)
        x = jax.nn.relu(self.l1(t))
        x = jax.nn.relu(self.l2(x))
        return self.l3(x)  # shape (batch, out_dim)

# -----------------------
# Problem: system u' = f(t, u)
# Example: linear system u' = A u  (stable)
# -----------------------
out_dim = 3
A = jnp.array([[-1.0, -0.2, 0.0],
               [0.1, -0.5, 0.0],
               [0.0, 0.2, -0.3]])  # example matrix

def f_true(t, u):
    # u shape (..., out_dim)
    return (u @ A.T)  # shape (..., out_dim)

u0 = jnp.array([1.0, 0.5, -0.2])  # desired initial condition vector

# -----------------------
# Init model and optimizers
# -----------------------
key = jax.random.PRNGKey(0)
model = MLP(key, in_dim=1, hidden=128, out_dim=out_dim)

# model optimizer
model_lr = 1e-3
model_opt = optax.adam(model_lr)
model_opt_state = model_opt.init(model.parameters())

# optional lambda optimizer (if you prefer a learned update rather than explicit AL step)
use_lambda_optimizer = False
lambda_lr = 1e-2
if use_lambda_optimizer:
    lambda_opt = optax.sgd(lambda_lr)
    lambda_opt_state = lambda_opt.init(u0)  # lambda has same shape as u0
else:
    lambda_opt = None
    lambda_opt_state = None

# augmented Lagrangian scalars
mu = 10.0             # initial penalty
mu_increase_factor = 2.0
mu_check_every = 200  # steps between mu-schedule checks
mu_plateau_tol = 1e-4  # threshold for "not improved" on |h|

# lambda multiplier (vector)
lam = jnp.zeros_like(u0)

# -----------------------
# Utilities: du/dt via jacobian, batched
# -----------------------
def time_to_model_input(ts):
    # ensure ts is shape (N,1)
    ts = jnp.asarray(ts)
    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)
    return ts

def dudt_batch(model, ts):
    # ts shape (N,1), returns dudt shape (N, out_dim)
    # compute jacobian w.r.t scalar t: for a scalar input -> vector output, jacrev gives (out_dim,)
    def single_dudt(t_single):
        # t_single shape (1,)
        # model expects shape (1,1) to preserve batch dimension
        out = model(t_single.reshape(1,1)).squeeze(0)  # (out_dim,)
        # jacrev over the scalar t -> returns (out_dim,) (d(out_i)/dt)
        return jax.jacrev(lambda s: model(s.reshape(1,1)).squeeze(0))(t_single)
    return jax.vmap(single_dudt)(ts.squeeze(-1))  # (N, out_dim)

# -----------------------
# Losses
# -----------------------
def residual_loss(model, ts):
    ts = time_to_model_input(ts)
    us = model(ts)[:, :]  # (N, out_dim)
    dudt = dudt_batch(model, ts)  # (N, out_dim)
    resid = dudt - f_true(ts.squeeze(-1)[:, None], us)  # broadcasting t if needed
    return jnp.mean(jnp.sum(resid**2, axis=-1))  # MSE over components

def constraint_value(model):
    t0 = jnp.array([[0.0]])
    return model(t0).squeeze(0)  # shape (out_dim,)

# -----------------------
# Augmented Lagrangian objective (for gradients w.r.t model params)
# -----------------------
def augmented_loss_for_params(params_like, model_obj, lam, mu, ts):
    # update model with params_like and evaluate
    m_tmp = model_obj.update_parameters(params_like)
    res = residual_loss(m_tmp, ts)
    h = constraint_value(m_tmp) - u0  # shape (out_dim,)
    aug = jnp.dot(lam, h) + 0.5 * mu * jnp.dot(h, h)
    return res + aug, (res, h)

# -----------------------
# Training step (jit compiled)
# Note: model is passed in and returns updated model.
# -----------------------
@jax.jit
def train_step(model, model_opt_state, lam, mu, ts, use_lambda_optimizer=False, lambda_opt_state=None):
    params = model.parameters()
    # compute loss and grads wrt params
    (loss_val, (res_val, h_val)), grads = jax.value_and_grad(augmented_loss_for_params, has_aux=True)(
        params, model, lam, mu, ts)

    # update model params
    updates, model_opt_state = model_opt.update(grads, model_opt_state, params)
    new_params = optax.apply_updates(params, updates)
    model = model.update_parameters(new_params)

    # lambda update options
    if use_lambda_optimizer:
        # perform a gradient-ascent step on lambda (we want to maximize Lagrangian w.r.t lambda),
        # so we apply gradient ASCENT: lambda += lr * grad_lambda (here grad_lambda = h)
        # But optax works for minimizing, so we minimize -lambda·h -> gradient is -h.
        # We'll compute lambda update via optax on the objective -lambda·h (i.e. grad = -h).
        # Build gradient manually and apply one optimizer step.
        # (Simpler approach: use explicit update outside jitted function; but we include a tiny opt step here)
        lambda_grads = -h_val  # gradient of (-lam · h) wrt lam
        lambda_updates, lambda_opt_state = lambda_opt.update(lambda_grads, lambda_opt_state, lam)
        lam = optax.apply_updates(lam, lambda_updates)
    else:
        # standard AL explicit update
        lam = lam + mu * h_val

    metrics = {
        "loss": loss_val,
        "residual_loss": res_val,
        "h": h_val,
    }
    return model, model_opt_state, lam, metrics, lambda_opt_state

# -----------------------
# Validation function (un-jitted for easy printing)
# -----------------------
def validate(model, ts_val):
    res = float(residual_loss(model, ts_val))
    h = jnp.array(constraint_value(model) - u0)
    return res, np.array(h)

# -----------------------
# Training loop with batching & mu scheduling
# -----------------------
rng = np.random.default_rng(123)
n_steps = 5000
batch_size = 64
t_domain = (0.0, 1.0)

best_h_norm = 1e9
steps_since_improve = 0

for step in range(1, n_steps+1):
    # sample collocation times uniformly in domain
    ts_batch = rng.uniform(t_domain[0], t_domain[1], size=(batch_size, 1)).astype(np.float32)

    model, model_opt_state, lam, metrics, lambda_opt_state = train_step(
        model, model_opt_state, lam, mu, ts_batch, use_lambda_optimizer=use_lambda_optimizer,
        lambda_opt_state=lambda_opt_state)

    if step % 100 == 0:
        # compute constraint violation norm
        h = np.linalg.norm(np.array(metrics["h"]), ord=2)
        res_val = float(metrics["residual_loss"])
        print(f"step {step:5d}  loss {metrics['loss']:.3e}  resid {res_val:.3e}  |h| {h:.3e}  mu {mu:.3e}")

    # mu scheduling: check every mu_check_every steps if |h| improved
    if step % mu_check_every == 0:
        curr_h_norm = float(jnp.linalg.norm(constraint_value(model) - u0))
        if curr_h_norm + mu_plateau_tol >= best_h_norm:
            # not improving -> increase mu
            mu = mu * mu_increase_factor
            print(f"  + mu increased to {mu:.3e} at step {step} (h {curr_h_norm:.3e} best {best_h_norm:.3e})")
            steps_since_improve = 0
        else:
            best_h_norm = curr_h_norm
            steps_since_improve = 0

# final validation
ts_val = np.linspace(0.0, 1.0, 200).reshape(-1,1).astype(np.float32)
res_val, h_vec = validate(model, ts_val)
print("FINAL residual:", res_val, "final constraint h:", h_vec, "||h||:", np.linalg.norm(h_vec))
print("predicted u(0):", np.array(constraint_value(model)))
print("target u0     :", np.array(u0))