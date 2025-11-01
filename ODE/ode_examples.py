import jax.numpy as jnp
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
        self.coeffs = jnp.stack([jnp.array(c) for c in self.coefs])  # shape (ncomp, degree+1)
        u0 = jnp.array([p(t_begin) for p in self.p_list])

        # self.p = np.polynomial.Polynomial(coef=coef)
        # self.dp = self.p.deriv(1)
        # self.coeffs = jnp.array(coef)
        # u0 = jnp.full(self.ncomp, self.p(t_begin))
        super().__init__(u0=u0, t_begin=t_begin, t_end=t_end)

    def a_coef(self, t):
        return jnp.zeros((self.ncomp, self.ncomp))

    def b_coef(self, t):
        # Evaluate each derivative dp_i(t)
        vals = jnp.array([jnp.polyval(self.coeffs[i, ::-1][1:], t) * (self.coeffs.shape[1] - 1 - i)
                          for i in range(self.ncomp)])
        # But we can also do it directly using dp_list:
        vals = jnp.array([p(t) for p in self.dp_list])
        return vals

    def solution(self, t):
        t = jnp.asarray(t)
        t = jnp.atleast_1d(t)
        # Evaluate each polynomial at t
        vals = jnp.stack([jnp.polyval(self.coeffs[i, ::-1], t)
                          for i in range(self.ncomp)], axis=-1)
        return vals  # shape: (len(t), ncomp)
    # def a_coef(self, t): return jnp.zeros(shape=(self.ncomp,self.ncomp))
    # def b_coef(self, t):
    #     return jnp.full(shape=(self.ncomp), fill_value=self.dp(t))
    # def solution(self, t):
    #     val = jnp.polyval(self.coeffs[::-1], t)
    #     val = jnp.atleast_1d(val)
    #     return jnp.repeat(val[:, None], self.ncomp, axis=1)
#-------------------------------------------------------------
class Exponential(OdeExample):
    def __init__(self, u0=2.3, t_begin=0.0, t_end=1.0, lam=1.1):
        super().__init__(u0, t_begin, t_end)
        self.lam = lam
    def f(self, u): return self.lam*u
    def a_coef(self, t): return jnp.array([[self.lam]])
    def b_coef(self, t): return jnp.array([0.0])
    def solution(self, t):
        return self.u0*jnp.exp(self.lam*t)

#-------------------------------------------------------------
class ExponentialJordan(OdeExample):
    def __init__(self, u0=jnp.array([1.0,1.0]), t_begin=0.0, t_end=3.0, lam=0.4):
        super().__init__(u0, t_begin, t_end)
        self.lam = lam
        self.A = jnp.array([[self.lam, 1.0], [0.0, self.lam]])
    def f(self, u):
        return self.A@u
    def a_coef(self, t): return self.A
    def b_coef(self, t): return jnp.array([0.0,0.0])
    def solution(self, t):
        exp = jnp.exp(self.lam*t)
        return jnp.array([exp*(1+t), exp]).T

#-------------------------------------------------------------
class TimeDependentRotation(OdeExample):
    def __init__(self, omega=lambda t: 1 + 0.5*jnp.sin(t),
                 u0=jnp.array([1.0, 0.0]), t_begin=0.0, t_end=12.0):
        super().__init__(u0, t_begin, t_end)
        self.omega = omega

    def a_coef(self, t):
        w = self.omega(t)
        return jnp.array([[0.0, -w], [w, 0.0]])

    def b_coef(self, t): return jnp.zeros(2)

    def solution(self, t):
        theta = t + 0.5- 0.5*jnp.cos(t)  # numerical approx ok for smooth ω
        # return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
        return jnp.array([jnp.cos(theta), jnp.sin(theta)]).T

#-------------------------------------------------------------
class RotScaleForce(OdeExample):
    def __init__(self, u0=jnp.array([1.0, 0.0]), t_begin=0.0, t_end=10.0):
        print(f"{self.u_exact(0)=}")
        super().__init__(u0, t_begin, t_end)
        # Example: rotation + scaling
        self.alpha = lambda t: 0.1 * t          # scaling rate
        self.beta  = lambda t: 1 + 0.5*jnp.sin(t)  # angular velocity

    def r(self, t):
        return jnp.exp(0.05 * t ** 2)

    def theta(self, t):
        return t + 0.5*(1 - jnp.cos(t))  # integral of beta(t)

    def u_exact(self, t):
        t = jnp.atleast_1d(t).astype(jnp.float64)
        theta = self.theta(t)
        scale = self.r(t)[..., None]
        u_rot = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1) * scale
        # forcing term shifted to satisfy u(0) = u0
        f = jnp.stack([jnp.sin(t), jnp.cos(t)], axis=-1)
        f = f - f[0]  # shift to match initial condition
        u = u_rot + f
        if u.shape[0] == 1:
            return u[0]
        return u
        theta = self.theta(t)
        scale = self.r(t)[..., None]
        # scale = jnp.exp(self.alpha(t) * t)[..., None]  # add extra axis for broadcasting
        f = jnp.stack([jnp.sin(t), jnp.cos(t)], axis=-1)
        return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1) * scale + f

    def a_coef(self, t):
        return jnp.array([[self.alpha(t), -self.beta(t)],
                          [self.beta(t),  self.alpha(t)]])

    def b_coef(self, t):
        t = jnp.asarray(t, dtype=jnp.float64)  # ensure floating point
        du_dt = jax.jacrev(self.u_exact)(t)
        return du_dt - self.a_coef(t) @ self.u_exact(t)

    def solution(self, t):
        return self.u_exact(t)
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
        return self.G / (1.0 + jnp.exp(-self.k*self.G*t)*(self.G/self.u0 - 1.0))

#-------------------------------------------------------------
class Pendulum(OdeExample):
    def __init__(self, alpha = 0.05, goL = 5.0, t_end=10, is_linear=True):
        super().__init__(u0=jnp.array([0.8 * jnp.pi, 0.0]), t_end=t_end)
        self.alpha, self.goL = alpha, goL
        self.f = self.f_linear if is_linear else self.f_nonlinear
        self.df = self.df_linear if is_linear else self.df_nonlinear
    def f_linear(self, u):
        theta, omega = u[0], u[1]
        alpha, goL = self.alpha, self.goL
        return jnp.array([omega, -alpha * omega - goL * theta])
    def df_linear(self, u):
        alpha, goL = self.alpha, self.goL
        return jnp.array([[0, 1], [-goL, -alpha]])
    def f_nonlinear(self, u):
        theta, omega = u[0], u[1]
        alpha, goL = self.alpha, self.goL
        return jnp.array([omega, -alpha * omega - goL * jnp.sin(theta)])
    def df_nonlinear(self, u):
        alpha, goL = self.alpha, self.goL
        return jnp.array([[0, 1], [-goL* jnp.cos(u[0]), -alpha]])

#-------------------------------------------------------------
class Lorenz(OdeExample):
    def __init__(self, sigma=10, rho=28, beta=8/3, t_end=20):
        super().__init__(u0=jnp.array([-10, -4.45, 35.1]), t_end=t_end)
        self.FP1 =  [jnp.sqrt(beta*(rho-1)), jnp.sqrt(beta*(rho-1)),rho-1]
        self.FP2 =  [-jnp.sqrt(beta*(rho-1)), -jnp.sqrt(beta*(rho-1)),rho-1]
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.f = lambda u: jnp.array([self.sigma*(u[1]-u[0]), self.rho*u[0]-u[1]-u[0]*u[2], u[0]*u[1]-self.beta*u[2]])
        self.df = lambda u: jnp.array([[-self.sigma, self.sigma,0], [self.rho-u[2],-1,-u[0]], [u[1],u[0],-self.beta]])
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

