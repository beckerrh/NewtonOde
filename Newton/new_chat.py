import jax.numpy as jnp
from jax import grad


def damped_newton(f, x0, max_iter=10, tol=1e-8, alpha=0.3, beta=0.5):
    """
    Solve f(x)=0 using damped Newton with backtracking.

    Parameters
    ----------
    f : callable
        Scalar function f(x)
    x0 : float
        Initial guess
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance |f(x)|
    alpha : float
        Armijo constant for backtracking (0 < alpha < 0.5)
    beta : float
        Step length reduction factor (0 < beta < 1)

    Returns
    -------
    x : float
        Approximate root
    """
    x = x0
    df = grad(lambda x: jnp.squeeze(f(x)))  # derivative of f
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(fx) < tol:
            return x
        # Newton step
        s = fx / dfx
        # Backtracking line search
        lam = 1.0
        phi0 = 0.5 * fx ** 2
        while 0.5 * (f(x - lam * s)) ** 2 > phi0 - alpha * lam * fx * s:
            lam *= beta
        x = x - lam * s
        print(f"Iteration {i}:  {x.item()=} {f(x).item()=} {s.item()=} {dfx.item()=} {lam=}")
    raise RuntimeError("Newton did not converge")

f = lambda x: 10.0 * jnp.sin(2.0 * x) + 4.0 - x**2

x_root = damped_newton(f, x0=3.0)
print("Root:", x_root)
print("f(root):", f(x_root))