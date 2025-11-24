import jax

import numpy as np
import os
from types import SimpleNamespace

#----------------------------------------------------------------------
class NewtonDriverNumpy:
    def __init__(self, **kwargs):
        if 'test_problem' in kwargs:
            test_problem = kwargs['test_problem']
            self.F = test_problem.F
            self.dF = test_problem.dF
        else:
            self.F = kwargs.pop('F')
            self.dF = kwargs.pop('dF')
    def computeResidual(self, x):
        xnorm = np.linalg.norm(x)
        r = self.F(x)
        meritvalue = 0.5*np.sum(r**2)
        resnorm = np.sqrt(2.0*meritvalue)
        return SimpleNamespace(residual=r, meritvalue=meritvalue, residual_norm=resnorm, x_norm=xnorm)
    def computeUpdate(self, **kwargs):
        r = kwargs['r']
        x = kwargs['x']
        J = np.atleast_2d(self.dF(x))
        try:
            dx = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            print(f"{J=}")
            assert None
        return SimpleNamespace(update=dx, update_norm=np.linalg.norm(dx), meritgrad=-np.sum(r**2))
    def computeUpdateSimple(self, **kwargs):
        r = kwargs['r']
        x = kwargs['x']
        J = np.atleast_2d(self.dF(x))
        dx = -J.T @ r
        return SimpleNamespace(update=dx, update_norm=np.linalg.norm(dx), meritgrad=-np.sum(dx**2))


#----------------------------------------------------------------------
class TestProblem:
    def __init__(self, x0=None, F=None, xS = None):
        self.name = self.__class__.__name__
        if not hasattr(self, 'x0'):
            if x0 is not None:
                self.x0 = np.asarray(x0)
                self.ncomp = self.x0.size
        else:
            self.ncomp = self.x0.size
        if not hasattr(self, 'xS') and xS is not None: self.xS = np.asarray(xS)
        if not hasattr(self, 'F'): self.F = F
        import jax
        jax.config.update("jax_enable_x64", True)
        self.dF = jax.jacrev(self.F)
        # if os.getenv("BACKEND", "numpy").lower() == "jax":
        #     self.dF = jax.jacrev(self.F)
        # else:
        #     if self.ncomp == 1:
        #         x = sp.symbols('x0')
        #         F_sym = [self.F(x)]   # wrap scalar in a list
        #         f = sp.Matrix(F_sym)
        #         J = f.jacobian([x])
        #         self.dF = sp.lambdify([x], J, 'numpy')
        #     else:
        #         x = sp.symbols(f'x0:{self.ncomp}')
        #         F_sym = self.F(x)
        #         f = sp.Matrix(F_sym)
        #         J = f.jacobian(x)
        #         self.dF = sp.lambdify([x], J, 'numpy')

class Simple_Quadratic(TestProblem):
    def __init__(self): super().__init__(x0=3.0, F=lambda x:(x-1.0)**2, xS=1.0)

class Cubic_Inflection(TestProblem):
    def __init__(self): super().__init__(x0=0.0, F=lambda x:x**3-2.0*x+2.0)
class Oscillatory(TestProblem):
    def __init__(self): super().__init__(x0=1.0, F=lambda x:math.cos(x) + x)
class Sine_Polynomial(TestProblem):
    def __init__(self): super().__init__(x0=3.0, F=lambda x:10*math.sin(2*x) + 4.0 - x**2)
class Exponential(TestProblem):
    def __init__(self): super().__init__(x0=3.0, F=lambda x:math.exp(x)-10*x)
#multiple roots
class Rosenbrock(TestProblem):
    def __init__(self):
        super().__init__(x0=[-1.2, 1.0], F=lambda x:np.array([10*(x[1]-x[0]**2), 1-x[0]]),
                xS=[1.0,1.0])
#textbook example to test globalization strategies and handling of singular Jacobians
class Powell_Singular(TestProblem):
    def __init__(self):
        self.x0 = np.array([3.0,-1.0,0.0,1.0])
        self.xS = np.array([0.0,0.0,0.0,0.0])
        self.sq5, self.sq10 = np.sqrt(5), np.sqrt(10)
        super().__init__()
    def F(self, x):
        return np.array([x[0] + 10*x[1], self.sq5 * (x[2] - x[3]), (x[1] - 2*x[2])**2, self.sq10 * (x[0] - x[3])**2])
class BroydenTridiag(TestProblem):
    def __init__(self, n=3):
        self.x0 = -1.0*np.ones(n)
        super().__init__()
    def F(self, x):
        # pad with zeros at both ends to simplify indexing
        x_pad = np.concatenate([np.array([0.0]), x, np.array([0.0])])
        xi = x_pad[1:-1]
        xim1 = x_pad[:-2]
        xip1 = x_pad[2:]
        return (3.0 - 2.0 * xi) * xi - xim1 - 2.0 * xip1 + 1.0
class CircleExponential(TestProblem):
    def __init__(self, n = 10, a=1.0, b=1.0, c=1.0, d=-1.0, value=1.0):
        self.x0 = value * np.ones(n)
        self.a, self.b, self.c, self.d = a, b, c, d
        self.F = self.F_jax if os.getenv("BACKEND", "numpy").lower() == "jax" else self.F_np
        super().__init__()
    def F_jax(self, x):
        x_im1 = np.roll(x, 1)  # x_{i-1}, with x_0 <- x_n
        x_ip1 = np.roll(x, -1)  # x_{i+1}, with x_{n+1} <- x_1
        return self.a * math.exp(x) - self.b * x_im1 - self.c * x_ip1 + self.d
    def F_np(self, x):
        xlist = list(x)
        return np.array([
            self.a * math.exp(xlist[i])
            - self.b * xlist[i - 1]  # cyclic i-1
            - self.c * xlist[(i + 1) % self.ncomp]  # cyclic i+1
            + self.d
            for i in range(self.ncomp)
        ])
class BumpExample(TestProblem):
    def bump(self, s):
        # smooth compact bump: e^{-1/(1-s^2)} for |s|<1 else 0
        y = np.zeros_like(s)
        mask = np.abs(s) < 1
        z = 1 - (s[mask])**2
        y[mask] = np.exp(-1.0 / z)
        return y

    def f(self, x, eps=1e-2, J=10):
        # scalar function
        y = 0.5 * (x-1.0)**2
        for j in range(1, J+1):
            y += eps * 2.0**(-j) * self.bump(2.0**j * (x-1.0))
        return y

