import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jaxopt

t0, t1, n_colloc, n_machine = 0, 3, 100, 30
t_colloc = np.linspace(t0, t1, n_colloc)

def u(t):
    return np.sin(np.pi * t)

def relu_hat_basis(t, t_machine):
    t_machine = jnp.sort(t_machine)
    n_machine = len(t_machine)

    # Ghost points
    t_left = t_machine[0] - (t_machine[1]-t_machine[0])
    t_right = t_machine[-1] + (t_machine[-1]-t_machine[-2])
    t_full = jnp.concatenate([jnp.array([t_left]), t_machine, jnp.array([t_right])])

    dt = t_full[1:] - t_full[:-1]
    dtinv = 1.0 / dt

    # Make alpha, beta, gamma exactly length n_machine
    alpha = dtinv[1:1+n_machine]   # slice of length 10
    gamma = dtinv[0:n_machine]     # slice of length 10
    beta = -alpha - gamma

    W = jnp.ones(len(t_full))
    b = -t_full

    y = jnp.maximum(0.0, W[:, None] * t[None,:] + b[:, None])

    # Now all slices have length n_machine
    phi = alpha[:, None] * y[2:2+n_machine,:] \
        + beta[:, None] * y[1:1+n_machine,:] \
        + gamma[:, None] * y[0:0+n_machine,:]

    return phi   # shape (n_machine, len(t))
def model(params, t):
    phi = relu_hat_basis(t, params["t_machine"])   # (n_machine, n_points)
    return jnp.dot(params["weights"], phi)

def loss(params):
    res = model(params, t_colloc) - u(t_colloc)
    return jnp.mean(res**2)

params = {
    "weights": jnp.zeros(n_machine),
    "t_machine": jnp.linspace(0, 3, n_machine)     # trainable mesh!
}

solver = jaxopt.LBFGS(fun=loss, maxiter=200, tol=1e-8)
res = solver.run(params)
params = res.params
print("Final Loss:", res.state.value)
print("Optimized weights:", params["weights"])
print("Optimized machine points:", params["t_machine"])


t_plot = np.linspace(t0, t1, 100)
plt.plot(t_plot, u(t_plot), label='u')
plt.plot(t_plot, np.array(model(params, t_plot)), label='u_app')
plt.plot(t_colloc, np.zeros_like(t_colloc), '-r', label='t_colloc')
plt.plot( params["t_machine"], np.zeros_like( params["t_machine"]), 'ok', label='t_machine')
plt.legend()
plt.xlabel("t")
plt.ylabel("u(t)")
plt.grid(True)
plt.show()
