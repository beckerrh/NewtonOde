import jax.numpy as jnp

#==================================================================
class OdeExample:
    """
    Examples for ODE
    du/dt = f(u)
    u(t_begin) = u0
    """
    def __init__(self, u0, t_begin=0.0, t_end=1.0):
        self.t_begin, self.t_end, self.u0 = t_begin, t_end, u0
        self.name = self.__class__.__name__
        self.ncomp = 1 if isinstance(self.u0, float) else len(self.u0)

#-------------------------------------------------------------
class Exponential(OdeExample):
    def __init__(self, u0=2.3, t_begin=0.0, t_end=1.0, lam=1.1):
        super().__init__(u0, t_begin, t_end)
        self.lam = lam
    def f(self, u):
        return self.lam*u
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
    def solution(self, t):
        exp = jnp.exp(self.lam*t)
        return jnp.array([exp*(1+t), exp]).T
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

