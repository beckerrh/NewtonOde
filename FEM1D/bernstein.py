from types import SimpleNamespace
import numpy as np
from scipy.special import binom

def bernstein_basis(k, x):
    # assert k > 0
    x = np.asarray(x)
    i = np.arange(k + 1)[:, None]
    return binom(k, i) * x[None, :]**i * (1 - x[None, :])**(k - i)

def bernstein_derivative(k, x):
    x = np.asarray(x)
    assert k > 0
    Bm1 = bernstein_basis(k - 1, x)
    dB = np.zeros((k + 1, len(x)))
    dB[0] = -k * Bm1[0]
    dB[k] = k * Bm1[k - 1]
    dB[1:k] = k * (Bm1[:-1] - Bm1[1:])
    return dB

def bernstein_second_derivative(k, x):
    x = np.asarray(x)
    d2B = np.zeros((k + 1, len(x)))
    if k < 2:
        return d2B
    Bm2 = bernstein_basis(k - 2, x)
    for i in range(k + 1):
        if i - 2 >= 0:
            d2B[i] += k * (k - 1) * Bm2[i - 2]
        if 0 <= i - 1 <= k - 2:
            d2B[i] += -2 * k * (k - 1) * Bm2[i - 1]
        if i <= k - 2:
            d2B[i] += k * (k - 1) * Bm2[i]
    return d2B

def reference_integration(degree, k):
    q, w = np.polynomial.legendre.leggauss(degree)
    # map [-1,1] -> [0,1]
    xi = 0.5 * (q + 1.0)
    w = 0.5 * w

    phi = bernstein_basis(k, xi)
    dphi = bernstein_derivative(k, xi)
    d2phi = bernstein_second_derivative(k, xi)

    return SimpleNamespace(
        xi=xi,
        w=w,
        n=len(xi),
        phi=phi,
        dphi=dphi,
        d2phi=d2phi,
    )

# -------------------------------------------------------------
def subdivide_bernstein_midpoint(c):
    """
    Subdivide Bernstein coefficients on [0,1] into coefficients
    on [0,1/2] and [1/2,1].

    Convention:
        B_i^k(x) = C(k,i) x^i (1-x)^{k-i}
    so c[0] is the left endpoint and c[k] is the right endpoint.
    """
    k = c.shape[0] - 1

    levels = [c.copy()]
    for r in range(1, k + 1):
        prev = levels[-1]
        levels.append(0.5 * (prev[:-1, :] + prev[1:, :]))

    cleft = np.zeros_like(c)
    cright = np.zeros_like(c)
    for i in range(k + 1):
        cleft[i, :] = levels[i][0, :]
        cright[i, :] = levels[k - i][i, :]

    return cleft, cright
