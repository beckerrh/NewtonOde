import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

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
        return np.zeros((self.ncomp, self.ncomp))

    def b_coef(self, t):
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
class TimeDependentRotation(OdeExample):
    def __init__(self, omega=lambda t: 1 + 0.5*np.sin(t),
                 u0=np.array([1.0, 0.0]), t_begin=0.0, t_end=12.0):
        super().__init__(u0, t_begin, t_end)
        self.omega = omega

    def a_coef(self, t):
        w = self.omega(t)
        return np.array([[0.0, -w], [w, 0.0]])

    def b_coef(self, t): return np.zeros(2)

    def solution(self, t):
        theta = t + 0.5- 0.5*np.cos(t)  # numerical approx ok for smooth ω
        # return np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        return np.array([np.cos(theta), np.sin(theta)]).T

#-------------------------------------------------------------
class RotScaleForce(OdeExample):
    def __init__(self, u0=np.array([1.0, 0.0]), t_begin=0.0, t_end=10.0):
        super().__init__(u0, t_begin, t_end)
        # Example: rotation + scaling
        self.alpha = lambda t: 0.1 * t             # scaling rate
        self.beta  = lambda t: 1 + 0.5*np.sin(t)  # angular velocity

    def r(self, t):
        """Scaling factor"""
        return np.exp(0.05 * t**2)

    def theta(self, t):
        """Angle (integral of beta)"""
        return t + 0.5*(1 - np.cos(t))

    def u_exact(self, t):
        t = np.atleast_1d(t)  # ensure array
        theta = self.theta(t)
        scale = self.r(t)[..., None]  # shape (nt,1)
        u_rot = np.stack([np.cos(theta), np.sin(theta)], axis=-1) * scale  # shape (nt,2)
        f = np.stack([np.sin(t), np.cos(t)], axis=-1)  # shape (nt,2)
        f = f - f[0]  # shift to satisfy initial condition
        u = u_rot + f  # shape (nt,2)
        return u  # always 2D
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

    def b_coef(self, t):
        """Vectorized b_coef over array t"""
        t = np.atleast_1d(t)
        u = np.atleast_2d(self.u_exact(t))  # shape (nq, ncomp)
        a = self.a_coef(t)           # (nq,2,2)
        # Compute du/dt manually
        alpha = self.alpha(t)
        beta  = self.beta(t)
        du_dt = np.zeros_like(u)
        du_dt[:,0] = alpha*u[:,0] - beta*u[:,1]
        du_dt[:,1] = beta*u[:,0] + alpha*u[:,1]
        return du_dt - np.einsum('qij,qj->qi', a, u)  # (nq,2)

    def solution(self, t):
        return self.u_exact(t)
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

