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
#-------------------------------------------------------------
class LogOscillatory(OdeExample):
    def __init__(self, x0=np.array([1.0,0.0]), t_begin=0.0, t_end=1.0):
        super().__init__(x0, t_begin, t_end)
        self.t0 = 0.5*(t_begin + t_end)

    def omega(self, t):
        return np.cos(np.log(np.abs(t - self.t0)))

    def theta(self, t):
        x  = t - self.t0
        x0 = 0 - self.t0
        def prim(z):
            return 0.5*z*(np.sin(np.log(np.abs(z))) +
                           np.cos(np.log(np.abs(z))))
        return prim(x) - prim(x0)

    def df(self, t, u):
        w = self.omega(t)
        return np.array([[0, -w], [w, 0]])

    def f(self, t, u):
        return self.df(t,u) @ u

    def solution(self, t):
        th = self.theta(t)
        return np.column_stack([np.cos(th), np.sin(th)])

#-------------------------------------------------------------
class ArctanJump(OdeExample):
    def __init__(self, t_begin=0.0, t_end=1.0,
                 alpha=10.0, eps=1e-2):
        self.t0 = 0.5*(t_begin + t_end)
        self.alpha = alpha
        self.eps = eps
        super().__init__(self.solution(t_begin), t_begin, t_end)

    def theta(self, t):
        return t + self.alpha * np.arctan((t - self.t0)/self.eps)

    def omega(self, t):
        return 1 + self.alpha*self.eps / ((t - self.t0)**2 + self.eps**2)

    def df(self, t, u):
        w = self.omega(t)
        return np.array([[0, -w], [w, 0]])

    def f(self, t, u):
        return self.df(t,u) @ u

    def solution(self, t):
        th = self.theta(t)
        return np.array([np.cos(th), np.sin(th)]).T
        return np.column_stack([np.cos(th), np.sin(th)])#==================Nonlinear examples============================


# -------------------------------------------------------------
class LogFrequency(OdeExample):
    def __init__(self, x0=np.array([1.0, 0.0]),
                 t_begin=1e-6, t_end=1.0, alpha=10.0):
        super().__init__(x0, t_begin, t_end)
        self.alpha = alpha

    def omega(self, t):
        return np.cos(self.alpha * np.log(t))

    def theta(self, t):
        a = self.alpha
        C = 1.0 / (1 + a * a)

        def prim(x):
            return C * x * (np.cos(a * np.log(x)) + a * np.sin(a * np.log(x)))

        return prim(t) - prim(1.0)  # normalize so theta(1)=0

    def df(self, t, u):
        w = self.omega(t)
        return np.array([[0, -w], [w, 0]])

    def f(self, t, u):
        return self.df(t, u) @ u

    def solution(self, t):
        th = self.theta(t)
        return np.column_stack([np.cos(th), np.sin(th)])

# -------------------------------------------------------------
class LinearPBInstability(OdeExample):

    def __init__(self, x0=np.array([1.0, 0.0]), t_begin=0.0, t_end=5.0):
        super().__init__(x0, t_begin, t_end)

    def omega(self, t):
        return t * np.cos(t**2)

    def df(self, t, u):
        w = self.omega(t)
        return np.array([[0.0, -w],
                         [w,   0.0]])

    def f(self, t, u):
        return self.df(t, u) @ u

    def theta(self, t):
        # Exact primitive: integral of s*cos(s^2) = 0.5*sin(s^2)
        return 0.5 * np.sin(t**2)

    def solution(self, t):
        th = self.theta(t)
        return np.array([np.cos(th), np.sin(th)]).T
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

    def plot(self, t, u):
        import matplotlib.pyplot as plt
        theta, omega = u[0], u[1]
        L = 1.0  # geometric length for plotting
        x = L * np.sin(theta)
        y = -L * np.cos(theta)
        plt.figure()
        plt.plot(x, y, lw=0.7)
        plt.scatter([x[0], x[-1]], [y[0], y[-1]],
                       c=["green", "red"], s=50, label="start / end")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()
#-------------------------------------------------------------
class DoublePendulum(OdeExample):
    def __init__(self, m1=1.3, m2=1.0, l1=1.0, l2=1.3,
                 g=9.81, t_end=5.0, x0=[np.pi/2, 0.0, np.pi/2+0.1, 0.0]):
        """
        State vector u = [theta1, omega1, theta2, omega2].
        Angles measured from vertical downward.
        """
        super().__init__(x0=np.array(x0), t_end=t_end)
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.g  = g
        # Shorthand for equations
    def f(self, t,u):
        th1, w1, th2, w2 = u
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        delta = th1 - th2
        denom = 2*m1 + m2 - m2 * np.cos(2*delta)
        ddth1 = (
            -g*(2*m1 + m2) * np.sin(th1)
            - m2*g*np.sin(th1 - 2*th2)
            - 2*np.sin(delta)*m2 * (w2**2 * l2 + w1**2 * l1 * np.cos(delta))
        ) / (l1 * denom)
        ddth2 = (
            2*np.sin(delta) * (
                w1**2 * l1 * (m1 + m2)
                + g*(m1 + m2)*np.cos(th1)
                + w2**2 * l2 * m2 * np.cos(delta)
            )
        ) / (l2 * denom)
        return np.array([w1, ddth1, w2, ddth2])

    def df(self, t, u):
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        th1, w1, th2, w2 = u
        delta = th1 - th2

        s1 = np.sin(th1)
        c1 = np.cos(th1)
        sd = np.sin(delta)
        cd = np.cos(delta)
        # s1m2 = np.sin(th1 - 2 * th2)
        c1m2 = np.cos(th1 - 2 * th2)
        # s2d = np.sin(2 * delta)

        D = (2 * m1 + m2 - m2 * np.cos(2 * delta))  # denom
        # To avoid accidental division by zero in degenerate configurations:
        if abs(D) < 1e-14:
            D = np.sign(D) * 1e-14 if D != 0 else 1e-14

        # Reconstruct numerators N1 and N2 (same as in the rhs)
        N1 = (-g * (2 * m1 + m2) * np.sin(th1)
              - m2 * g * np.sin(th1 - 2 * th2)
              - 2 * sd * m2 * (w2 ** 2 * l2 + w1 ** 2 * l1 * cd))

        N2 = (2 * sd *
              (w1 ** 2 * l1 * (m1 + m2)
               + g * (m1 + m2) * c1
               + w2 ** 2 * l2 * m2 * cd))

        # Partial derivatives wrt omega1, omega2 (only through N1,N2)
        # dN1/dw1, dN1/dw2
        dN1_dw1 = -4 * m2 * l1 * cd * sd * w1  # derivative of -2*sd*m2*(w1^2*l1*cd + ...)
        dN1_dw2 = -4 * m2 * l2 * sd * w2

        # dN2/dw1, dN2/dw2
        dN2_dw1 = 4 * (m1 + m2) * l1 * sd * w1
        dN2_dw2 = 4 * m2 * l2 * sd * cd * w2

        # Derivatives of D wrt theta1, theta2
        dD_dth1 = 2 * m2 * np.sin(2 * delta)  # = 2 m2 sin(2Δ)
        dD_dth2 = -2 * m2 * np.sin(2 * delta)  # = -2 m2 sin(2Δ)

        # Partial derivatives of N1 wrt theta1, theta2
        # computed from manual differentiation (see thread for derivation)
        # dN1/dth1:
        #   = -g(2m1+m2) cos(th1) - m2 g cos(th1-2th2)
        #     - 2 m2 [ w2^2 l2 cosΔ + w1^2 l1 cos(2Δ) ]
        dN1_dth1 = (-g * (2 * m1 + m2) * c1
                    - m2 * g * c1m2
                    - 2 * m2 * (w2 ** 2 * l2 * cd + w1 ** 2 * l1 * np.cos(2 * delta)))

        # dN1/dth2:
        #   =  2 m2 g cos(th1-2th2) + 2 m2 [ w2^2 l2 cosΔ + w1^2 l1 cos(2Δ) ]
        dN1_dth2 = (2 * m2 * g * c1m2
                    + 2 * m2 * (w2 ** 2 * l2 * cd + w1 ** 2 * l1 * np.cos(2 * delta)))

        # Partial derivatives of N2 wrt theta1, theta2
        # Using N2 = 2 sinΔ * A, A = w1^2 l1 (m1+m2) + g (m1+m2) cos(th1) + w2^2 l2 m2 cosΔ
        A = (w1 ** 2 * l1 * (m1 + m2)
             + g * (m1 + m2) * c1
             + w2 ** 2 * l2 * m2 * cd)

        # dN2/dth1 = 2[ cosΔ * A + sinΔ * ( - g (m1+m2) sin(th1) - w2^2 l2 m2 sinΔ ) ]
        dN2_dth1 = (2 * (cd * A
                         - sd * (g * (m1 + m2) * s1 + w2 ** 2 * l2 * m2 * sd)))

        # dN2/dth2 = 2[ -cosΔ * A + sinΔ * ( w2^2 l2 m2 sinΔ ) ]
        dN2_dth2 = (2 * (-cd * A + sd * (w2 ** 2 * l2 * m2 * sd)))

        # Now construct Jacobian J (4x4)
        J = np.zeros((4, 4))

        # f1 = omega1
        J[0, :] = [0.0, 1.0, 0.0, 0.0]
        # f3 = omega2
        J[2, :] = [0.0, 0.0, 0.0, 1.0]

        # f2 = ddth1 = N1 / (l1 * D)
        # Partial wrt w1, w2
        J[1, 1] = dN1_dw1 / (l1 * D)
        J[1, 3] = dN1_dw2 / (l1 * D)
        # Partial wrt th1
        J[1, 0] = (dN1_dth1 * D - N1 * dD_dth1) / (l1 * D * D)
        # Partial wrt th2
        J[1, 2] = (dN1_dth2 * D - N1 * dD_dth2) / (l1 * D * D)

        # f4 = ddth2 = N2 / (l2 * D)
        # Partial wrt w1, w2
        J[3, 1] = dN2_dw1 / (l2 * D)
        J[3, 3] = dN2_dw2 / (l2 * D)
        # Partial wrt th1
        J[3, 0] = (dN2_dth1 * D - N2 * dD_dth1) / (l2 * D * D)
        # Partial wrt th2
        J[3, 2] = (dN2_dth2 * D - N2 * dD_dth2) / (l2 * D * D)

        return J
    def plot(self, t, u):
        import matplotlib.pyplot as plt
        # u has shape (4, len(t)); unpack directly
        th1, w1, th2, w2 = u
        # Cartesian coordinates for animation / phase inspection
        x1 = self.l1 * np.sin(th1)
        y1 = -self.l1 * np.cos(th1)
        x2 = x1 + self.l2 * np.sin(th2)
        y2 = y1 - self.l2 * np.cos(th2)
        fig, axs = plt.subplots(1, 3, figsize=(16, 4))

        # -----------------------------------------------------------
        # (1) Physical trajectory
        axs[0].plot(x1, y1, lw=0.7, label="mass 1 path")
        axs[0].plot(x2, y2, lw=0.7, label="mass 2 path")
        axs[0].scatter([x1[0], x2[0]], [y1[0], y2[0]], c='g', label="start")
        axs[0].scatter([x1[-1], x2[-1]], [y1[-1], y2[-1]], c='r', label="end")
        axs[0].set_aspect('equal', adjustable='box')
        axs[0].set_title("Trajectory")
        axs[0].legend()
        axs[0].grid(True)

        # -----------------------------------------------------------
        # (2) Phase portrait for the first pendulum
        axs[1].plot(th1, w1, lw=0.7)
        axs[1].scatter(th1[0], w1[0], c='g')
        axs[1].scatter(th1[-1], w1[-1], c='r')
        axs[1].set_title(r"Phase Portrait $(\theta_1,\omega_1)$")
        axs[1].set_xlabel(r"$\theta_1$")
        axs[1].set_ylabel(r"$\omega_1$")
        axs[1].grid(True)

        # -----------------------------------------------------------
        # (3) Phase portrait for the second pendulum
        axs[2].plot(th2, w2, lw=0.7)
        axs[2].scatter(th2[0], w2[0], c='g')
        axs[2].scatter(th2[-1], w2[-1], c='r')
        axs[2].set_title(r"Phase Portrait $(\theta_2,\omega_2)$")
        axs[2].set_xlabel(r"$\theta_2$")
        axs[2].set_ylabel(r"$\omega_2$")
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()
#-------------------------------------------------------------
class Lorenz(OdeExample):
    def __init__(self, sigma=10.0, rho=28.0, beta=8/3, t_end=20.0, x0=[-10, -4.45, 35.1]):
        super().__init__(x0=np.array(x0), t_end=t_end)
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

    def plot(self, t, u):
        import matplotlib.pyplot as plt
        x, dxdt = u[0], u[1]
        plt.figure(figsize=(8, 6))
        plt.plot(x, dxdt, lw=0.7, label="Phase Portrait (Limit Cycle)")
        plt.title("Van der Pol Oscillator - Limit Cycle")
        plt.xlabel("x(t)")
        plt.ylabel("dx/dt")
        plt.legend()
        plt.grid(True)
        plt.show()

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

