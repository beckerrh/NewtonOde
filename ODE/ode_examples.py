import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import sympy as sp

#==================================================================
class OdeExample:
    """
    Examples for ODE
    dx/dt = f(t,x)
    x(t_begin) = x0

    du/dt = a_coef u + b_coef
    """
    def __init__(self, x0, t_begin=0.0, t_end=1.0):
        self.t_begin, self.t_end, self.x0 = t_begin, t_end, x0
        self.name = self.__class__.__name__
        self.ncomp = 1 if isinstance(self.x0, float) else len(self.x0)

    def _create_time_derivative(self, sol_func):
        """
        Given solution u(t), return a function u'(t) using SymPy.
        Works for vector-valued solutions.
        """
        t_sym = sp.symbols('t')
        # Evaluate solution symbolically
        u_sym = sol_func(t_sym)  # should return sympy expressions
        # If scalar, wrap as list
        if not isinstance(u_sym, (list, tuple, sp.Matrix)):
            u_sym = [u_sym]
        u_sym = sp.Matrix(u_sym)
        # Compute derivative
        u_dot_sym = u_sym.diff(t_sym)
        # Lambdify into numpy function
        u_dot_func = sp.lambdify(t_sym, u_dot_sym, modules='numpy')

        def u_dot_wrapped(t_num):
            val = u_dot_func(t_num)
            val = np.array(val)

            # Handle scalar t
            if val.ndim == 1:
                val = val[np.newaxis, :]  # (1, ncomp)
            # Handle (ncomp, nt) → (nt, ncomp)
            elif val.shape[0] == self.ncomp and val.shape[1] == t_num.shape[0]:
                val = val.T
            # Handle (1, ncomp, nt) from SymPy
            elif val.ndim == 3 and val.shape[0] == 1 and val.shape[1] == self.ncomp and val.shape[2] == t_num.shape[0]:
                val = val[0].T  # -> (nt, ncomp)
            return val
        return u_dot_wrapped
    def check_linear(self):
        if hasattr(self, 'solution') and not hasattr(self, 'solution_t'):
            self.solution_t = self._create_time_derivative(self.solution)
        t_test = np.linspace(0, 2, 5)
        u = self.solution(t_test)
        b = self.b_coef(t_test)
        a = self.a_coef(t_test)
        residual = np.einsum('tij,tj->ti', a, u) + b - self.solution_t(t_test)
        ok = np.allclose(residual,np.zeros_like(residual))
        if not ok:
            print(residual)  # should be very small
        return ok

#=============================================================
#==================Linear examples============================
#=============================================================
class LinearIntegration(OdeExample):
    def __init__(self, t_begin=0.0, t_end=1.0):
        super().__init__(x0=1.1, t_begin=t_begin, t_end=t_end)
        self.f = lambda t,u: 2.2
        self.df = lambda t,u: np.array([[0.0]])
    def solution(self, t):
        t = np.atleast_1d(t)
        return 1.1+2.2*t
#-------------------------------------------------------------
class PolynomialIntegration(OdeExample):
    def __init__(self, t_begin=0.0, t_end=1.0, degree=2, seed=None, ncomp=1):
        self.ncomp = ncomp
        rng = np.random.default_rng(seed)
        self.coefs = [rng.uniform(-1.0, 1.0, degree + 1) for _ in range(ncomp)]
        self.p_list = [np.polynomial.Polynomial(c) for c in self.coefs]
        self.dp_list = [p.deriv(1) for p in self.p_list]
        self.coeffs = np.stack([np.array(c) for c in self.coefs])  # shape (ncomp, degree+1)
        x0 = np.array([p(t_begin) for p in self.p_list])
        super().__init__(x0=x0, t_begin=t_begin, t_end=t_end)
    def df(self, t, u):
        t = np.atleast_1d(t)
        return np.zeros((t.size, self.ncomp, self.ncomp))
    def f(self, t, u):
        # t = np.atleast_1d(t)
        vals = np.array([p(t) for p in self.dp_list])
        return vals
    def solution(self, t):
        t = np.asarray(t)
        t = np.atleast_1d(t)
        # Evaluate each polynomial at t
        vals = np.stack([np.polyval(self.coeffs[i, ::-1], t)
                          for i in range(self.ncomp)], axis=-1)
        return vals  # shape: (len(t), ncomp)
#-------------------------------------------------------------
class Exponential(OdeExample):
    def __init__(self, x0=2.3, t_begin=0.0, t_end=1.0, lam=1.1):
        super().__init__(x0, t_begin, t_end)
        self.lam = lam
    def f(self, t, u): return self.lam*u
    def df(self, t, u): return np.array([[self.lam]])
    # def b_coef(self, t): return np.array([0.0])
    def solution(self, t):
        return np.stack([self.x0*np.exp(self.lam*t)], axis=-1)

#-------------------------------------------------------------
class ExponentialJordan(OdeExample):
    def __init__(self, x0=np.array([1.0,1.0]), t_begin=0.0, t_end=3.0, lam=0.4):
        super().__init__(x0, t_begin, t_end)
        self.lam = lam
        self.A = np.array([[self.lam, 1.0], [0.0, self.lam]])
    def f(self, t, u):
        return self.A@u
    def df(self, t, u): return self.A
    # def b_coef(self, t): return np.array([0.0,0.0])
    def solution(self, t):
        exp = np.exp(self.lam*t)
        return np.stack([exp * (1 + t), exp], axis=-1)

#-------------------------------------------------------------
class TimeDependentShear(OdeExample):
    def __init__(self):
        super().__init__(t_begin=0.0, t_end=4.0, x0=self.solution(0.0).squeeze())
    def df(self, t, u):
        return np.array([[-1.0, np.sin(t)],[np.cos(t),-1.0]])
    def solution(self, t):
        e, c, s = np.exp(-t), np.cos(t), np.sin(t)
        return np.stack([e * c, e * s], axis=-1)  # (N,2)
    def f(self, t, u):
        e, c, s = np.exp(-t), np.cos(t), np.sin(t)
        return np.array([-e*(s+s**2), e*(c-c**2)])+self.df(t,u)@u
 #-------------------------------------------------------------
class TimeDependentRotation(OdeExample):
    def __init__(self, omega=lambda t: 1 + 0.5*np.sin(t),
                 x0=np.array([1.0, 0.0]), t_begin=0.0, t_end=12.0):
        super().__init__(x0, t_begin, t_end)
        self.omega = omega
    def df(self, t, u):
        w = self.omega(t)
        return np.array([[0,-w],[w,0]])
    def f(self, t, u):
        A = self.df(t, u)
        return A@u
    def solution(self, t):
        theta = t + 0.5- 0.5*np.cos(t)
        return np.array([np.cos(theta), np.sin(theta)]).T
#-------------------------------------------------------------
class LinearSingular(OdeExample):
    def __init__(self, x0=np.array([1.0, 0.0]), t_begin=0.0, t_end=1.0):
        super().__init__(x0, t_begin, t_end)
        self.ts = (self.t_end-self.t_begin)/2.0
    def g(self,t): return np.cos(np.log(np.abs(t-self.ts)))
    def theta(self,t):
        ts = self.ts
        x = t - ts
        ell = np.log(np.abs(x))
        prim = 0.5 * x * (np.sin(ell) + np.cos(ell))
        # enforce initial condition θ(0)=0
        x0 = 0 - ts
        ell0 = np.log(np.abs(x0))
        prim0 = 0.5 * x0 * (np.sin(ell0) + np.cos(ell0))
        return prim - prim0
    def df(self, t, u):
        w = self.g(t)
        return np.array([[0,-w],[w,0]])
    def f(self, t, u):
        A = self.df(t, u)
        return A@u
    def solution(self, t):
        th = self.theta(t)
        return np.array([np.cos(th), np.sin(th)]).T

#=============================================================
#==================Nonlinear examples============================
#=============================================================
#-------------------------------------------------------------
class Logistic(OdeExample):
    def __init__(self, k=1.5, G=1.0, x0=0.5, t_end=
4):
        super().__init__(x0=x0, t_end=t_end)
        self.k, self.G = k, G
    def f(self, t, u):
        return self.k * u * (self.G - u)
    def df(self, t, u):
        return self.k * (self.G - 2*u)
    def solution(self, t):
        return self.G / (1.0 + np.exp(-self.k*self.G*t)*(self.G/self.x0 - 1.0))

#-------------------------------------------------------------
class Pendulum(OdeExample):
    def __init__(self, alpha = 0.05, goL = 5.0, t_end=10, is_linear=False):
        super().__init__(x0=np.array([0.8 * np.pi, 0.0]), t_end=t_end)
        self.alpha, self.goL = alpha, goL
        self.f = self.f_linear if is_linear else self.f_nonlinear
        self.df = self.df_linear if is_linear else self.df_nonlinear
    def f_linear(self, t, u):
        theta, omega = u[0], u[1]
        alpha, goL = self.alpha, self.goL
        return np.array([omega, -alpha * omega - goL * theta])
    def df_linear(self, t, u):
        alpha, goL = self.alpha, self.goL
        return np.array([[0, 1], [-goL, -alpha]])
    def f_nonlinear(self, t, u):
        theta, omega = u[0], u[1]
        alpha, goL = self.alpha, self.goL
        return np.array([omega, -alpha * omega - goL * np.sin(theta)])
    def df_nonlinear(self, t, u):
        alpha, goL = self.alpha, self.goL
        return np.array([[0, 1], [-goL* np.cos(u[0]), -alpha]])

#-------------------------------------------------------------
class LorenzOld(OdeExample):
    def __init__(self, sigma=10, rho=28, beta=8/3, t_end=20):
        super().__init__(x0=np.array([-10, -4.45, 35.1]), t_end=t_end)
        self.FP1 =  [np.sqrt(beta*(rho-1)), np.sqrt(beta*(rho-1)),rho-1]
        self.FP2 =  [-np.sqrt(beta*(rho-1)), -np.sqrt(beta*(rho-1)),rho-1]
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.f = lambda t,u: np.array([self.sigma*(u[1]-u[0]), self.rho*u[0]-u[1]-u[0]*u[2], u[0]*u[1]-self.beta*u[2]])
        self.df = lambda t,u: np.array([[-self.sigma, self.sigma,0], [self.rho-u[2],-1,-u[0]], [u[1],u[0],-self.beta]])
    def plot(self, t, u):
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d')
        x,y,z = u[0], u[1], u[2]
        # x,y,z = u[:,0], u[:,1], u[:,2]
        ax.plot(x, y, z, label='u', lw=0.5)
        ax.plot(x[-1], y[-1], z[-1], 'X', label="u(T)")
        ax.plot(x[0], y[0], z[0], 'X', label="u(0)")
        ax.plot(*self.FP1, color='k', marker="8", ls='')
        ax.plot(*self.FP2, color='k', marker="8", ls='')
        ax.view_init(26, 130)
        ax.legend()
        plt.show()
#-------------------------------------------------------------
class VanDerPol(OdeExample):
    """Van der Pol oscillator: u'' - μ(1-u²)u' + u = 0"""
    def __init__(self, mu=10.0, x0=[2.0, 0.0], t_end=40.0):
        super().__init__(x0=np.array(x0), t_end=t_end)
        self.mu = mu
    def f(self, t, u):
        return np.array([u[1], self.mu*(1 - u[0]**2)*u[1] - u[0]])
    def df(self, t, u):
        return np.array([[0.0, 1.0],
                         [-1 - 2*self.mu*u[0]*u[1], self.mu*(1 - u[0]**2)]])
    # no closed-form solution
#-------------------------------------------------------------
class Robertson(OdeExample):
    """Classic stiff chemical kinetics system"""
    def __init__(self, x0=[1.0, 0.0, 0.0], t_end=1e3):
        super().__init__(x0=np.array(x0), t_end=t_end)
        # self.k1, self.k2, self.k3 = 0.04, 1e4, 3e7
        self.k1, self.k2, self.k3 = 0.04, 3e7, 1e4

    def f(self, t, u):
        y1, y2, y3 = u
        return np.array([
            -self.k1*y1 + self.k3*y2*y3,
             self.k1*y1 - self.k2*y2*y2 - self.k3*y2*y3,
             self.k2*y2*y2
        ])
    def df(self, t, u):
        y1, y2, y3 = u
        return np.array([
            [-self.k1,         self.k3*y3,       self.k3*y2],
            [ self.k1, -2*self.k2*y2 - self.k3*y3, -self.k3*y2],
            [ 0.0,             2*self.k2*y2,       0.0]
        ])    # no analytic solution
#-------------------------------------------------------------
class Lorenz(OdeExample):
    """Lorenz chaotic system"""
    def __init__(self, sigma=10.0, rho=28.0, beta=8/3, x0=[1.0, 1.0, 1.0], t_end=
40.0):
        super().__init__(x0=np.array(x0), t_end=t_end)
        self.sigma, self.rho, self.beta = sigma, rho, beta
    def f(self, t, u):
        x, y, z = u
        return np.array([self.sigma*(y-x), x*(self.rho - z)-y, x*y - self.beta*z])
    def df(self, t, u):
        x, y, z = u
        return np.array([
            [-self.sigma, self.sigma, 0.0],
            [self.rho - z, -1.0, -x],
            [y, x, -self.beta]
        ])
    # no analytic solution
#-------------------------------------------------------------
class Mathieu(OdeExample):
    """Mathieu equation: u'' + (a + b cos t) u = 0"""
    def __init__(self, a=1.0, b=0.5, x0=[1.0, 0.0], t_end=
100.0):
        super().__init__(x0=np.array(x0), t_end=t_end)
        self.a, self.b = a, b
    def f(self, t, u):
        return np.array([u[1], -(self.a + self.b*np.cos(t)) * u[0]])
    def df(self, t, u):
        return np.array([[0.0, 1.0], [-(self.a + self.b*np.cos(t)), 0.0]])
#-------------------------------------------------------------
class NonlinearMix(OdeExample):
    """Mixed nonlinear system: du1/dt = -u2 + u1(1-u1²), du2/dt = u1 + u2(1-u2²)"""
    def __init__(self, x0=[1.0, 0.0], t_end=
10.0):
        super().__init__(x0=np.array(x0), t_end=t_end)
    def f(self, t, u):
        u1, u2 = u
        return np.array([-u2 + u1*(1-u1**2), u1 + u2*(1-u2**2)])
    def df(self, t, u):
        u1, u2 = u
        return np.array([
            [1 - 3*u1**2, -1.0],
            [1.0, 1 - 3*u2**2]
        ])

