import numpy as np

#-------------------------------------------------------------
class EllipticExample:
    def __init__(self, x0, x1, uL=None, uR=None):
        self.name = self.__class__.__name__
        self.x0, self.x1 = x0, x1

        if uL is None:
            assert uR is None
            assert hasattr(self, 'solution')
            uL = self.solution(x0)
            uR = self.solution(x1)

        uL = np.atleast_1d(uL).astype(float)
        uR = np.atleast_1d(uR).astype(float)

        assert uL.shape == uR.shape

        self.ncomp = len(uL)
        self.uL = uL
        self.uR = uR

    def df_du(self, x, u):
        """
        x : (ne, nq)
        u : (ne, nq, ncomp)

        returns:
            (ne, nq, ncomp, ncomp)
        """
        ne, nq, ncomp = u.shape
        return np.zeros((ne, nq, ncomp, ncomp), dtype=u.dtype)

    def dsolution(self, x):
        """
        x: scalar or array
        returns same shape as solution(x), or (..., ncomp)
        """
        raise NotImplementedError

    def diffusion_coef(self, x):
        return np.ones_like(x)
    # for estimator
    def diffusion_coef_x(self, x):
        return np.zeros_like(x)

    def convection_coef(self, x):
        return np.zeros_like(x)

#-------------------------------------------------------------
class Poisson(EllipticExample):
    def __init__(self):
        super().__init__(x0=0, x1=1)
    def f(self, x, u):
        # rhs = np.ones_like(x)
        rhs = 9*np.cos(3*x)
        return rhs[..., None]
    def solution(self, x):
        # u = 0.5*x*(1-x)
        u = np.cos(3*x) - (1-x)*np.cos(0) - x * np.cos(3)
        return u[..., None]
    def dsolution(self, x):
        # du = 0.5*(1-2*x)
        du = -3*np.sin(3*x) + np.cos(0) - np.cos(3)
        return du[..., None]


#-------------------------------------------------------------
class OscillatoryPoisson(EllipticExample):
    def __init__(self, omega=5, alpha=3.0):
        self.omega = omega
        self.alpha = alpha
        super().__init__(x0=0.0, x1=1.0)

    def solution(self, x):
        w = self.omega * np.pi
        a = self.alpha
        u = np.sin(w*x) + x*(1-x)*np.exp(a*x)
        return u[..., None]

    def dsolution(self, x):
        w = self.omega * np.pi
        a = self.alpha
        du = w*np.cos(w*x) + np.exp(a*x) * (1 - 2*x + a*x*(1-x))
        return du[..., None]

    def f(self, x, u):
        w = self.omega * np.pi
        a = self.alpha
        rhs = w**2 * np.sin(w*x) - np.exp(a*x) * (-2 + 2*a*(1 - 2*x) + a**2*x*(1-x))
        return rhs[..., None]
#-------------------------------------------------------------
class DiscontinuousAlphaOscillator(EllipticExample):
    """
    Interface problem:

        -(alpha u')' = f

    with discontinuous alpha and prescribed flux

        q = alpha u'.

    Therefore

        f = -q',
        u' = q / alpha.

    The flux q is continuous and vanishes at the interface.
    """

    def __init__(
        self,
        m_left=2,
        m_right=5,
        jump=20.0,
        exp_left=0.0,
        exp_right=0.0,
        center_left=0.30,
        center_right=0.75,
    ):
        self.m_left = m_left
        self.m_right = m_right
        self.jump = jump
        self.exp_left = exp_left
        self.exp_right = exp_right
        self.center_left = center_left
        self.center_right = center_right
        super().__init__(x0=0.0, x1=1.0)

    def diffusion_coef(self, x):
        x = np.asarray(x)
        return np.where(x < 0.5, 1.0, self.jump)

    def flux(self, x):
        x = np.asarray(x)
        q = np.empty_like(x, dtype=float)

        left = x < 0.5
        yL = 2*x[left]
        yR = 2*x[~left] - 1

        q[left] = np.sin(2*np.pi*self.m_left*yL)
        q[~left] = np.sin(2*np.pi*self.m_right*yR)

        if self.exp_left > 0:
            q[left] *= np.exp(-self.exp_left*(x[left] - self.center_left)**2)

        if self.exp_right > 0:
            q[~left] *= np.exp(-self.exp_right*(x[~left] - self.center_right)**2)

        return q

    def dflux(self, x):
        x = np.asarray(x)
        dq = np.empty_like(x, dtype=float)

        left = x < 0.5
        yL = 2*x[left]
        yR = 2*x[~left] - 1

        # base sine parts
        sL = np.sin(2*np.pi*self.m_left*yL)
        cL = np.cos(2*np.pi*self.m_left*yL)
        sR = np.sin(2*np.pi*self.m_right*yR)
        cR = np.cos(2*np.pi*self.m_right*yR)

        dq[left] = 4*np.pi*self.m_left*cL
        dq[~left] = 4*np.pi*self.m_right*cR

        if self.exp_left > 0:
            E = np.exp(-self.exp_left*(x[left] - self.center_left)**2)
            dq[left] = E * (
                dq[left]
                - 2*self.exp_left*(x[left] - self.center_left)*sL
            )

        if self.exp_right > 0:
            E = np.exp(-self.exp_right*(x[~left] - self.center_right)**2)
            dq[~left] = E * (
                dq[~left]
                - 2*self.exp_right*(x[~left] - self.center_right)*sR
            )

        return dq

    def dsolution(self, x):
        return (self.flux(x) / self.diffusion_coef(x))[..., None]

    def solution(self, x):
        x = np.asarray(x)
        u = np.empty_like(x, dtype=float)

        left = x < 0.5
        ml = self.m_left
        mr = self.m_right
        J = self.jump

        yL = 2 * x[left]
        yR = 2 * x[~left] - 1

        u[left] = (1 - np.cos(2 * np.pi * ml * yL)) / (4 * np.pi * ml)
        u[~left] = (1 - np.cos(2 * np.pi * mr * yR)) / (J * 4 * np.pi * mr)

        return u[..., None]
    def f(self, x, u):
        return (-self.dflux(x))[..., None]
#-------------------------------------------------------------
class InteriorLayerVariableAlpha(EllipticExample):
    def __init__(self, lam=200.0, xc=0.37):
        self.lam = lam
        self.xc = xc
        super().__init__(0, 1)

    def dirichlet(self):
        return np.zeros(1), np.zeros(1)

    def diffusion_coef(self, x):
        return 1.0 + 0.5 * np.sin(7 * np.pi * x) + 0.2 * x
    def diffusion_coef_x(self, x):
        return 0.5*7*np.pi*np.cos(7*np.pi*x) + 0.2

    def solution(self, x):
        L, xc = self.lam, self.xc
        u = x*(1-x)*np.arctan(L*(x-xc))
        return u[..., None]

    def dsolution(self, x):
        L, xc = self.lam, self.xc
        z = L*(x-xc)
        du = (1-2*x)*np.arctan(z) + x*(1-x)*L/(1+z**2)
        return du[..., None]

    def ddsolution(self, x):
        L, xc = self.lam, self.xc
        z = L*(x-xc)
        ddu = -2*np.arctan(z) + 2*(1-2*x)*L/(1+z**2) - 2*x*(1-x)*L**2*z/(1+z**2)**2
        return ddu[..., None]

    def f(self, x, u=None):
        a = self.diffusion_coef(x)
        ax = self.diffusion_coef_x(x)
        return -ax[..., None] * self.dsolution(x) - a[..., None] * self.ddsolution(x)

# -------------------------------------------------------------
class LinearSystem3(EllipticExample):
    def __init__(self):
        super().__init__(0, 1, uL=self.solution(0.0), uR=self.solution(1.0))

    def solution(self, x):
        return np.stack([
            1.0 + x + np.sin(np.pi*x),
            -1.0 + 2.0*x + x*(1.0-x),
            0.5 - x + np.sin(2*np.pi*x),
        ], axis=-1)

    def dsolution(self, x):
        return np.stack([
            1.0 + np.pi*np.cos(np.pi*x),
            2.0 + 1.0 - 2.0*x,
            -1.0 + 2.0*np.pi*np.cos(2*np.pi*x),
        ], axis=-1)

    def ddsolution(self, x):
        return np.stack([
            -np.pi**2*np.sin(np.pi*x),
            -2.0*np.ones_like(x),
            -4.0*np.pi**2*np.sin(2*np.pi*x),
        ], axis=-1)

    def diffusion_coef(self, x):
        return np.array([
            [1.0, 0.15, 0.0],
            [0.15, 2.0, 0.20],
            [0.0, 0.20, 3.0],
        ])

    def convection_coef(self, x):
        return np.zeros((3, 3))

    def R(self):
        return np.array([
            [2.0, 0.5, 0.0],
            [0.5, 3.0, 0.25],
            [0.0, 0.25, 4.0],
        ])

    def df_du(self, x, u):
        R = self.R()
        return np.broadcast_to(R, (*x.shape, 3, 3))

    def f(self, x, u):
        d2uex = self.ddsolution(x)
        A = self.diffusion_coef(x)
        R = self.R()

        return (
                -np.einsum("ab,...b->...a", A, d2uex)
                - np.einsum("ab,...b->...a", R, self.solution(x))
                + np.einsum("ab,...b->...a", R, u)
        )