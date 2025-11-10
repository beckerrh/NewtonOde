import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import sympy as sp

#==================================================================
class OdeExample:
    """
    Examples for ODE
    du/dt = f(u)
    u(t_begin) = u0

    du/dt = a_coef u + b_coef
    """
    def __init__(self, u0, t_begin=0.0, t_end=1.0):
        self.t_begin, self.t_end, self.u0 = t_begin, t_end, u0
        self.name = self.__class__.__name__
        self.ncomp = 1 if isinstance(self.u0, float) else len(self.u0)
        if hasattr(self, 'f'):
            self.df = jax.jacrev(self.f)
            self.is_linear = False
        else:
            self.is_linear = True

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
#-------------------------------------------------------------
class PolynomialIntegration(OdeExample):
    def __init__(self, t_begin=0.0, t_end=1.0, degree=2, seed=None, ncomp=1):
        self.ncomp = ncomp
        rng = np.random.default_rng(seed)
        self.coefs = [rng.uniform(-1.0, 1.0, degree + 1) for _ in range(ncomp)]
        self.p_list = [np.polynomial.Polynomial(c) for c in self.coefs]
        self.dp_list = [p.deriv(1) for p in self.p_list]
        self.coeffs = np.stack([np.array(c) for c in self.coefs])  # shape (ncomp, degree+1)
        u0 = np.array([p(t_begin) for p in self.p_list])
        super().__init__(u0=u0, t_begin=t_begin, t_end=t_end)

    def a_coef(self, t):
        t = np.atleast_1d(t)
        return np.zeros((t.size, self.ncomp, self.ncomp))

    def b_coef(self, t):
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
    def __init__(self, u0=2.3, t_begin=0.0, t_end=1.0, lam=1.1):
        super().__init__(u0, t_begin, t_end)
        self.lam = lam
    def f(self, u): return self.lam*u
    def a_coef(self, t): return np.array([[self.lam]])
    def b_coef(self, t): return np.array([0.0])
    def solution(self, t):
        return np.stack([self.u0*np.exp(self.lam*t)], axis=-1)

#-------------------------------------------------------------
class ExponentialJordan(OdeExample):
    def __init__(self, u0=np.array([1.0,1.0]), t_begin=0.0, t_end=3.0, lam=0.4):
        super().__init__(u0, t_begin, t_end)
        self.lam = lam
        self.A = np.array([[self.lam, 1.0], [0.0, self.lam]])
    def f(self, u):
        return self.A@u
    def a_coef(self, t): return self.A
    def b_coef(self, t): return np.array([0.0,0.0])
    def solution(self, t):
        exp = np.exp(self.lam*t)
        return np.stack([exp * (1 + t), exp], axis=-1)

#-------------------------------------------------------------
class TimeDependentShear(OdeExample):
    def __init__(self):
        super().__init__(t_begin=0.0, t_end=4.0, u0=self.solution(0.0).squeeze())

    def a_coef(self, t):
        """Return A(t). For scalar t -> shape (2,2). For array t -> (nt,2,2)."""
        t = np.atleast_1d(t)
        n = t.shape[0]
        A = np.zeros((n, 2, 2))
        A[:, 0, 0] = -1.0
        A[:, 0, 1] = 1.0 + 0.5 * np.sin(t)
        A[:, 1, 0] = 0.2 * np.cos(t)
        A[:, 1, 1] = -2.0
        return A

    def solution(self, t):
        """Analytic u(t) = e^{-t}[cos t, sin t]; return shape (N,1,2)."""
        if isinstance(t, sp.Basic):
            exp = sp.exp
            cos = sp.cos
            sin = sp.sin
        else:
            exp = np.exp
            cos = np.cos
            sin = np.sin
        # t = np.atleast_1d(t)
        e = exp(-t)
        cos_t = cos(t)
        sin_t = sin(t)
        u = np.stack([e * cos_t, e * sin_t], axis=-1)  # (N,2)
        return u

    def b_coef(self, t):
        """
        Compute b(t) = u'(t) - A(t) u(t) exactly.
        Returns (2,) for scalar t, or (nt,2) for array t.
        """
        t = np.atleast_1d(t)
        n = t.shape[0]

        e = np.exp(-t)
        cos_t = np.cos(t)
        sin_t = np.sin(t)

        # u and u'
        u1 = e * cos_t
        u2 = e * sin_t
        u = np.stack([u1, u2], axis=-1)           # (n,2)

        u1_dot = e * (-cos_t - sin_t)
        u2_dot = e * (cos_t - sin_t)
        u_dot = np.stack([u1_dot, u2_dot], axis=-1)  # (n,2)

        # A(t)
        A = np.zeros((n,2,2))
        A[:,0,0] = -1.0
        A[:,0,1] = 1.0 + 0.5*np.sin(t)
        A[:,1,0] = 0.2*np.cos(t)
        A[:,1,1] = -2.0

        # compute A(t) @ u(t)
        Au = np.einsum('tij,tj->ti', A, u)  # (n,2)

        b = u_dot - Au
        return b
#-------------------------------------------------------------
class TimeDependentRotation(OdeExample):
    def __init__(self, omega=lambda t: 1 + 0.5*np.sin(t),
                 u0=np.array([1.0, 0.0]), t_begin=0.0, t_end=12.0):
        super().__init__(u0, t_begin, t_end)
        self.omega = omega

    def a_coef(self, t):
        w = np.atleast_1d(self.omega(t))
        A = np.zeros((w.size, 2, 2), dtype=float)
        A[:, 0, 1] = -w
        A[:, 1, 0] = w
        return np.squeeze(A)

    def b_coef(self, t):
        t = np.atleast_1d(t)
        return np.zeros(shape=(t.size,2))

    def solution(self, t):
        theta = t + 0.5- 0.5*np.cos(t)  # numerical approx ok for smooth ω
        return np.array([np.cos(theta), np.sin(theta)]).T

#-------------------------------------------------------------
class RotScaleForce(OdeExample):
    def __init__(self, t_end=2.0):
        # Example: rotation + scaling
        self.alpha = lambda t: 0.1 * t             # scaling rate
        self.beta  = lambda t: 1 + 0.5*np.sin(t)  # angular velocity
        self.alpha_t = lambda t: t             # scaling rate
        self.beta_t  = lambda t: 0.5*np.cos(t)  # angular velocity
        super().__init__(u0=self.solution(0.0).squeeze(), t_begin=0.0, t_end=t_end)
    def a_coef(self, t):
        """Vectorized a_coef over array t"""
        t = np.atleast_1d(t)
        alpha = self.alpha(t)
        beta  = self.beta(t)
        n = t.shape[0]
        a = np.zeros((n,2,2))
        a[:,0,0] = alpha
        a[:,0,1] = -beta
        a[:,1,0] = beta
        a[:,1,1] = alpha
        return a  # shape (n,2,2)


    def r(self, t):
        """Scaling factor"""
        if isinstance(t, sp.Basic):
            exp = sp.exp
            cos = sp.cos
            sin = sp.sin
        else:
            exp = np.exp
            cos = np.cos
            sin = np.sin
        return exp(0.05 * t**2)

    def theta(self, t):
        """Angle (integral of beta)"""
        if isinstance(t, sp.Basic):
            exp = sp.exp
            cos = sp.cos
            sin = sp.sin
        else:
            exp = np.exp
            cos = np.cos
            sin = np.sin
        return t + 0.5*(1 - cos(t))

    def solution(self, t):
        if isinstance(t, sp.Basic):
            exp = sp.exp
            cos = sp.cos
            sin = sp.sin
        else:
            exp = np.exp
            cos = np.cos
            sin = np.sin

        # t = np.atleast_1d(t)  # ensure array
        theta = self.theta(t)
        scale = self.r(t)  # shape (nt,1)
        # u_rot = np.stack([cos(theta), sin(theta)], axis=-1) * scale  # shape (nt,2)
        u_rot = np.array([scale * cos(theta), scale * sin(theta)]).T
        f = np.stack([sin(t), cos(t)], axis=-1)  # shape (nt,2)
        # f = f - f[0]  # shift to satisfy initial condition
        u = u_rot + f  # shape (nt,2)
        return u  # always 2D

    def b_coef(self, t):
        """
        Compute b(t) = u'(t) - A(t) u(t) exactly for the DG solver.

        Works for scalar t or array t, returns:
          - shape (2,) for scalar t
          - shape (nt,2) for array t
        """
        t = np.atleast_1d(t)
        nt = t.shape[0]

        # --- rotation + scaling
        theta = t + 0.5 * (1 - np.cos(t))  # integral of beta(t)
        theta_dot = 1 + 0.5 * np.sin(t)  # beta(t)
        scale = np.exp(0.05 * t ** 2)  # r(t)
        scale_dot = 0.1 * t * scale  # r'(t)

        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        # --- rotation term and derivative
        u_rot = np.stack([cos_th, sin_th], axis=-1) * scale[:, None]  # (nt,2)
        u_rot_dot = np.zeros_like(u_rot)
        u_rot_dot[:, 0] = scale_dot * cos_th - scale * sin_th * theta_dot
        u_rot_dot[:, 1] = scale_dot * sin_th + scale * cos_th * theta_dot

        # --- forcing term and derivative
        f = np.stack([np.sin(t), np.cos(t)], axis=-1) - np.array([0.0, 1.0])
        f_dot = np.stack([np.cos(t), -np.sin(t)], axis=-1)

        # --- total derivative
        u = u_rot + f
        u_dot = u_rot_dot + f_dot

        # --- A(t) @ u(t)
        alpha = 0.1 * t
        beta = 1 + 0.5 * np.sin(t)
        a = np.zeros((nt, 2, 2))
        a[:, 0, 0] = alpha
        a[:, 0, 1] = -beta
        a[:, 1, 0] = beta
        a[:, 1, 1] = alpha

        Au = np.einsum('tij,tj->ti', a, u)

        # --- b(t)
        b = u_dot - Au

        # --- return shape consistent with scalar input
        return b[0] if nt == 1 else b
    def b_coef2(self, t):
        t = np.atleast_1d(t)  # shape (nt,) or (1,)
        # u_exact returns (nt,2) for array t, or (2,) for scalar t
        u = np.atleast_2d(self.u_exact(t))  # (nt,2)

        # shape helpers
        alpha = self.alpha(t)  # (nt,)
        beta = self.beta(t)  # (nt,)
        scale = self.r(t)  # (nt,)
        scale_dot = (0.1 * t) * scale  # r'(t) = 0.1 t * exp(0.05 t^2)  == alpha*scale
        theta = self.theta(t)  # (nt,)
        theta_dot = 1 + 0.5 * np.sin(t)  # (nt,)

        # rotation vectors
        cos_th = np.cos(theta)  # (nt,)
        sin_th = np.sin(theta)  # (nt,)

        # u_rot = scale * [cos_th, sin_th]
        # compute derivative of u_rot:
        # u_rot_dot = scale_dot * [cos, sin] + scale * theta_dot * [-sin, cos]
        u_rot_dot = np.empty_like(u)  # (nt,2)
        u_rot_dot[:, 0] = scale_dot * cos_th - scale * sin_th * theta_dot
        u_rot_dot[:, 1] = scale_dot * sin_th + scale * cos_th * theta_dot

        # forcing derivative f'(t); original forcing was [sin t, cos t] minus shift
        f_dot = np.stack([np.cos(t), -np.sin(t)], axis=-1)  # (nt,2)

        # total time derivative of exact solution
        u_dot = u_rot_dot + f_dot  # (nt,2)

        # a(t) @ u(t) for all t
        a = self.a_coef(t)  # should return shape (nt,2,2)
        # a @ u  -> (nt,2)
        return u_dot - np.einsum('tij,tj->ti', a, u)

    # def solution(self, t):
    #     return self.u_exact(t)
# class RotScaleForce(OdeExample):
#     def __init__(self, u0=np.array([1.0, 0.0]), t_begin=0.0, t_end=10.0):
#         print(f"{self.u_exact(0)=}")
#         super().__init__(u0, t_begin, t_end)
#         # Example: rotation + scaling
#         self.alpha = lambda t: 0.1 * t          # scaling rate
#         self.beta  = lambda t: 1 + 0.5*np.sin(t)  # angular velocity
#
#     def r(self, t):
#         return np.exp(0.05 * t ** 2)
#
#     def theta(self, t):
#         return t + 0.5*(1 - np.cos(t))  # integral of beta(t)
#
#     def u_exact(self, t):
#         # t = np.atleast_1d(t).astype(np.float64)
#         theta = self.theta(t)
#         scale = self.r(t)[..., None]
#         u_rot = np.stack([np.cos(theta), np.sin(theta)], axis=-1) * scale
#         # forcing term shifted to satisfy u(0) = u0
#         f = np.stack([np.sin(t), np.cos(t)], axis=-1)
#         f = f - f[0]  # shift to match initial condition
#         u = u_rot + f
#         if u.shape[0] == 1:
#             return u[0]
#         return u
#
#     def a_coef(self, t):
#         return np.array([[self.alpha(t), -self.beta(t)],
#                           [self.beta(t),  self.alpha(t)]])
#
#     def b_coef(self, t):
#         t = np.asarray(t, dtype=np.float64)  # ensure floating point
#         du_dt = jax.jacrev(self.u_exact)(t)
#         return du_dt - self.a_coef(t) @ self.u_exact(t)
#
#     def solution(self, t):
#         return self.u_exact(t)
#=============================================================
#==================Nonlinear examples============================
#=============================================================
#-------------------------------------------------------------
class Logistic(OdeExample):
    def __init__(self, k=1.5, G=1, u0=0.5, T=2):
        super().__init__(u0=u0, t_end=T)
        self.k, self.G = k, G
    def f(self, u):
        return self.k * u * (self.G - u)
    def df(self, u):
        return self.k * (self.G - 2*u)
    def solution(self, t):
        return self.G / (1.0 + np.exp(-self.k*self.G*t)*(self.G/self.u0 - 1.0))

#-------------------------------------------------------------
class Pendulum(OdeExample):
    def __init__(self, alpha = 0.05, goL = 5.0, t_end=10, is_linear=True):
        super().__init__(u0=np.array([0.8 * np.pi, 0.0]), t_end=t_end)
        self.alpha, self.goL = alpha, goL
        self.f = self.f_linear if is_linear else self.f_nonlinear
        self.df = self.df_linear if is_linear else self.df_nonlinear
    def f_linear(self, u):
        theta, omega = u[0], u[1]
        alpha, goL = self.alpha, self.goL
        return np.array([omega, -alpha * omega - goL * theta])
    def df_linear(self, u):
        alpha, goL = self.alpha, self.goL
        return np.array([[0, 1], [-goL, -alpha]])
    def f_nonlinear(self, u):
        theta, omega = u[0], u[1]
        alpha, goL = self.alpha, self.goL
        return np.array([omega, -alpha * omega - goL * np.sin(theta)])
    def df_nonlinear(self, u):
        alpha, goL = self.alpha, self.goL
        return np.array([[0, 1], [-goL* np.cos(u[0]), -alpha]])

#-------------------------------------------------------------
class Lorenz(OdeExample):
    def __init__(self, sigma=10, rho=28, beta=8/3, t_end=20):
        super().__init__(u0=np.array([-10, -4.45, 35.1]), t_end=t_end)
        self.FP1 =  [np.sqrt(beta*(rho-1)), np.sqrt(beta*(rho-1)),rho-1]
        self.FP2 =  [-np.sqrt(beta*(rho-1)), -np.sqrt(beta*(rho-1)),rho-1]
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.f = lambda u: np.array([self.sigma*(u[1]-u[0]), self.rho*u[0]-u[1]-u[0]*u[2], u[0]*u[1]-self.beta*u[2]])
        self.df = lambda u: np.array([[-self.sigma, self.sigma,0], [self.rho-u[2],-1,-u[0]], [u[1],u[0],-self.beta]])
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

