#!/usr/bin/env python3
# -*- coding: utf-8 -*-

assert __name__ == '__main__'
import os
os.environ["BACKEND"] = 'jax'
import newton
from backend import np
import matplotlib.pyplot as plt
import jax

# df = lambda x: 20.0 * np.cos(2.0 * x) - 2.0 * x
f = lambda x: 10.0 * np.sin(2.0 * x) + 4.0 - x * x
df = jax.grad(lambda x: np.squeeze(f(x)))
x0 = [3.]
def cdx(r, x, info):
    J = df(x)
    # print(f"{J=} {r=} {x=} {f(x)=}")
    return r /J, 1, True


newton = newton.Newton(df=df)
# newton = newton.Newton(computedx=cdx, verbose_bt=True)
# newton.sdata.bt_maxiter = 40
# newton.sdata.bt_omega = 0.5
# newton.sdata.bt_c = 0.1
xs, info = newton.solve(x0, f, maxiter=50)
x = np.linspace(-1., 4.0)
plt.plot(x, f(x), [x[0], x[-1]], [0,0], '--r')
plt.plot(xs, f(xs), 'Xk')
plt.show()
