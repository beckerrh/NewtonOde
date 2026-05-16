from types import SimpleNamespace
import numpy as np
import sympy as sp

#=========================================================
def make_symbolic_problem(exprs, n):
    xs = sp.symbols(f"x0:{n}")
    exprs = sp.Matrix(exprs(xs))
    J = exprs.jacobian(xs)

    F_lamb = sp.lambdify((xs,), exprs, "numpy")
    J_lamb = sp.lambdify((xs,), J, "numpy")

    def F(x):
        return np.asarray(F_lamb(tuple(np.atleast_1d(x))), dtype=float).squeeze()

    def dF(x):
        return np.asarray(J_lamb(tuple(np.atleast_1d(x))), dtype=float)

    return F, dF

#=========================================================
class TestProblem:
    def __init__(self, x0, exprs=None, xS=None, F=None, dF=None):
        self.name = self.__class__.__name__

        self.x0 = np.atleast_1d(np.asarray(x0, dtype=float))
        self.ncomp = self.x0.size

        if xS is not None:
            self.xS = np.atleast_1d(np.asarray(xS, dtype=float))

        # symbolic variables
        xs = sp.symbols(f'x0:{self.ncomp}')

        if F is not None:
            assert dF is not None
            self.F = F
            self.dF = dF
        else:
            assert exprs is not None
            exprs = exprs(xs)
            if not isinstance(exprs, (list, tuple)):
                exprs = [exprs]

            Fsym = sp.Matrix(exprs)
            Jsym = Fsym.jacobian(xs)

            # numerical functions
            F_lamb = sp.lambdify((xs,), Fsym, "numpy")
            J_lamb = sp.lambdify((xs,), Jsym, "numpy")

            def F_num(x):
                x = np.asarray(x, dtype=float)

                # scalar problem evaluated on a plotting grid
                if self.ncomp == 1 and x.size > 1:
                    return np.array([F_num([xi]) for xi in x.ravel()]).reshape(x.shape)

                xx = np.atleast_1d(x)
                return np.asarray(
                    F_lamb(tuple(xx)),
                    dtype=float
                ).squeeze()
            self.F = F_num
            self.dF = lambda x: np.asarray(
                J_lamb(tuple(np.atleast_1d(x))),
                dtype=float
            )

#----------------------------------------------------------------------
class Simple_Quadratic(TestProblem):
    def __init__(self):
        super().__init__(
            x0=3.0,
            exprs=lambda x: (x[0]-1.0)**2,
            xS=1.0
        )
class Exponential(TestProblem):
    def __init__(self):
        super().__init__(
            x0=3.0,
            exprs=lambda x: sp.exp(x[0]) - 10*x[0]
        )
class Rosenbrock(TestProblem):
    def __init__(self):
        super().__init__(
            x0=[-1.2, 1.0],
            exprs=lambda x: [
                10*(x[1]-x[0]**2),
                1-x[0]
            ],
            xS=[1.0,1.0]
        )

import sympy as sp
import numpy as np

#=========================================================
class Cubic_Inflection(TestProblem):
    def __init__(self):
        super().__init__(
            x0=0.0,
            exprs=lambda x: x[0]**3 - 2.0*x[0] + 2.0
        )

#=========================================================
class Oscillatory(TestProblem):
    def __init__(self):
        super().__init__(
            x0=1.0,
            exprs=lambda x: sp.cos(x[0]) + x[0]
        )

#=========================================================
class Sine_Polynomial(TestProblem):
    def __init__(self):
        super().__init__(
            x0=3.0,
            exprs=lambda x: 10*sp.sin(2*x[0]) + 4.0 - x[0]**2
        )

#=========================================================
class Exponential(TestProblem):
    def __init__(self):
        super().__init__(
            x0=3.0,
            exprs=lambda x: sp.exp(x[0]) - 10*x[0]
        )

#=========================================================
# textbook example to test globalization strategies
# and handling of singular Jacobians
class Powell_Singular(TestProblem):
    def __init__(self):
        sq5 = np.sqrt(5)
        sq10 = np.sqrt(10)

        super().__init__(
            x0=[3.0, -1.0, 0.0, 1.0],
            exprs=lambda x: [
                x[0] + 10*x[1],
                sq5 * (x[2] - x[3]),
                (x[1] - 2*x[2])**2,
                sq10 * (x[0] - x[3])**2
            ],
            xS=[0.0, 0.0, 0.0, 0.0]
        )

#=========================================================
class BroydenTridiag(TestProblem):
    def __init__(self, n=3):

        def exprs(x):
            eqs = []

            for i in range(n):
                xim1 = x[i-1] if i > 0 else 0
                xip1 = x[i+1] if i < n-1 else 0

                eqs.append(
                    (3.0 - 2.0*x[i])*x[i]
                    - xim1
                    - 2.0*xip1
                    + 1.0
                )

            return eqs

        super().__init__(
            x0=-1.0*np.ones(n),
            exprs=exprs
        )

#=========================================================
class CircleExponential(TestProblem):
    def __init__(
        self,
        n=10,
        a=1.0,
        b=1.0,
        c=1.0,
        d=-1.0,
        value=1.0
    ):

        self.a, self.b, self.c, self.d = a, b, c, d

        def exprs(x):
            eqs = []

            for i in range(n):
                eqs.append(
                    a * sp.exp(x[i])
                    - b * x[i-1]
                    - c * x[(i+1) % n]
                    + d
                )

            return eqs

        super().__init__(
            x0=value*np.ones(n),
            exprs=exprs
        )
class BumpExample(TestProblem):
    def __init__(self, eps=1e-2, J=10):
        self.eps = eps
        self.J = J
        self.x0 = np.asarray([1.5])
        self.xS = np.asarray([1.0])
        self.name = self.__class__.__name__

        self.F = self.dphi
        self.dF = self.ddphi

    def bump(self, s):
        s = np.asarray(s, dtype=float)
        y = np.zeros_like(s)

        mask = np.abs(s) < 1.0
        z = 1.0 - s[mask]**2
        y[mask] = np.exp(-1.0 / z)

        return y

    def dbump(self, s):
        s = np.asarray(s, dtype=float)
        y = np.zeros_like(s)

        mask = np.abs(s) < 1.0
        z = 1.0 - s[mask]**2
        y[mask] = -2.0*s[mask] / z**2 * np.exp(-1.0 / z)

        return y

    def ddbump(self, s):
        s = np.asarray(s, dtype=float)
        y = np.zeros_like(s)

        mask = np.abs(s) < 1.0
        z = 1.0 - s[mask]**2
        b = np.exp(-1.0 / z)

        y[mask] = (
            (-2.0 / z**2 - 8.0*s[mask]**2 / z**3 + 4.0*s[mask]**2 / z**4)
            * b
        )

        return y

    def phi(self, x):
        x = np.asarray(x, dtype=float)
        y = 0.5 * (x - 1.0)**2

        for j in range(1, self.J + 1):
            s = 2.0**j * (x - 1.0)
            y += self.eps * 2.0**(-j) * self.bump(s)

        return y.squeeze()

    def dphi(self, x):
        x = np.asarray(x, dtype=float)
        y = x - 1.0

        for j in range(1, self.J + 1):
            s = 2.0**j * (x - 1.0)
            y += self.eps * self.dbump(s)

        return np.atleast_1d(y.squeeze())

    def ddphi(self, x):
        x = np.asarray(x, dtype=float)
        y = np.ones_like(x)

        for j in range(1, self.J + 1):
            s = 2.0**j * (x - 1.0)
            y += self.eps * 2.0**j * self.ddbump(s)

        return np.asarray([[y.squeeze()]], dtype=float)
