#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy.optimize import newton

assert __name__ == '__main__'
import newton, test_problems
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# df = lambda x: 20.0 * np.cos(2.0 * x) - 2.0 * x
f = lambda x: 10.0 * jnp.sin(2.0 * x) + 4.0 - x * x
df = jax.grad(lambda x: np.squeeze(f(x)))
x0 = np.array([3.])

nd = test_problems.NewtonDriverNumpy(F=f, dF=df)
xall=[]
nd.call_back = lambda x: xall.append(x)
newton = newton.Newton(nd=nd)
xs, info = newton.solve(x0, maxiter=50)
x = np.linspace(-1., 4.0)
plt.plot(x, f(x), [x[0], x[-1]], [0,0], '--r')
plt.plot(xs, f(xs), 'Xk')
plt.plot(x0, f(x0), 'Xm')
for i,x in enumerate(xall):
    plt.plot(x, f(x), 'o', label=rf"x_{i}")
plt.legend()
plt.show()
