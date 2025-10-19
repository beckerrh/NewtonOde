import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
np.random.seed(42)
n = 10       # Dimension
m = 20      # Number of terms
mu = 0.05   # Smoothing parameter
sigma = 1.0 # Regularization coefficient
M = 2 / mu  # Cubic parameter from paper
delta = 1e-6  # Damping for standard Newton
max_iters = 20

# Generate random data
A = np.random.randn(m, n)
b = np.random.randn(m)

# Define f_mu, gradient, Hessian
def f_mu(x):
    z = (A @ x - b) / mu
    z_max = np.max(z)
    return mu * (z_max + np.log(np.sum(np.exp(z - z_max))))

def grad_f_mu(x):
    z = (A @ x - b) / mu
    w = np.exp(z - np.max(z))
    w /= np.sum(w)
    return A.T @ w

def hess_f_mu(x):
    z = (A @ x - b) / mu
    w = np.exp(z - np.max(z))
    w /= np.sum(w)
    W = np.diag(w)
    return (A.T @ (W - np.outer(w, w)) @ A) / mu

# Gradient-Regularized Newton
def grad_reg_newton(x, sigma):
    g = grad_f_mu(x)
    H = hess_f_mu(x)
    beta = sigma * np.linalg.norm(g)
    step = np.linalg.solve(H + beta * np.eye(n), -g)
    return x + step

# Damped Newton
def damped_newton(x, delta):
    g = grad_f_mu(x)
    H = hess_f_mu(x)
    step = np.linalg.solve(H + delta * np.eye(n), -g)
    return x + step

# Cubic-Regularized Newton using gradient descent on the subproblem
def cubic_newton(x, M):
    g = grad_f_mu(x)
    H = hess_f_mu(x)

    def cubic_model(s):
        return g @ s + 0.5 * s @ H @ s + (M / 6) * np.linalg.norm(s)**3

    def grad_cubic_model(s):
        norm_s = np.linalg.norm(s)
        return g + H @ s + (0.5 * M * norm_s * s if norm_s > 1e-12 else 0)

    s = np.zeros_like(x)
    lr = 0.1
    for _ in range(20):
        s -= lr * grad_cubic_model(s)

    return x + s

# Initialize
x_gn = np.zeros(n)
x_std = np.zeros(n)
x_cub = np.zeros(n)

traj_gn = [f_mu(x_gn)]
traj_std = [f_mu(x_std)]
traj_cub = [f_mu(x_cub)]

# Run all methods
for _ in range(max_iters):
    x_gn = grad_reg_newton(x_gn, sigma)
    x_std = damped_newton(x_std, delta)
    x_cub = cubic_newton(x_cub, M)

    traj_gn.append(f_mu(x_gn))
    traj_std.append(f_mu(x_std))
    traj_cub.append(f_mu(x_cub))

# Normalize
min_val = min(traj_gn[-1], traj_std[-1], traj_cub[-1])
res_gn = np.array(traj_gn) - min_val
res_std = np.array(traj_std) - min_val
res_cub = np.array(traj_cub) - min_val

# Plot
plt.figure(figsize=(8, 5))
plt.semilogy(res_gn, label="Gradient-Regularized Newton", marker='o')
plt.semilogy(res_std, label="Damped Newton", marker='x')
plt.semilogy(res_cub, label="Cubic-Regularized Newton", marker='s')
plt.title("Comparison of Newton Variants on log-sum-exp")
plt.xlabel("Iteration")
plt.ylabel(r"$f_\mu(x_k) - \min f_\mu$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
