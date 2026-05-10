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

    def f(self, x, u):
        raise NotImplementedError

    def f_vec(self, x, u):
        """
        x: (ne, nq)
        u: (ne, nq, ncomp)
        returns: (ne, nq, ncomp)
        """
        return np.apply_along_axis(
            lambda z: self.f(z[0], z[1:]),
            axis=2,
            arr=np.concatenate([x[:, :, None], u], axis=2)
        )
    def dsolution(self, x):
        """
        x: scalar or array
        returns same shape as solution(x), or (..., ncomp)
        """
        raise NotImplementedError

    def p_coef(self, x):
        return 1.0

    def q_coef(self, x):
        return 0.0

    def r_coef(self, x):
        return 0.0

#-------------------------------------------------------------
class Poisson(EllipticExample):
    def __init__(self):
        super().__init__(x0=0, x1=1)
    def f_vec(self, x, u):
        # return np.ones_like(x)
        return 9*np.cos(3*x)
    def solution(self, x):
        # return 0.5*x*(1-x)
        return np.cos(3*x) - (1-x)*np.cos(0) - x * np.cos(3)
    def dsolution(self, x):
        # return 0.5*(1-2*x)
        return -3*np.sin(3*x) + np.cos(0) - np.cos(3)
#-------------------------------------------------------------
class SmoothPoisson(EllipticExample):
    def __init__(self):
        super().__init__(x0=0.0, x1=1.0)

    def solution(self, x):
        return np.sin(2*np.pi*x) + x*(1-x)*np.exp(x)

    def dsolution(self, x):
        return (
            2*np.pi*np.cos(2*np.pi*x)
            + (1 - x - x**2)*np.exp(x)
        )

    def f_vec(self, x, u):
        return (
            4*np.pi**2*np.sin(2*np.pi*x)
            + (3*x + x**2)*np.exp(x)
        )
#-------------------------------------------------------------
class OscillatoryPoisson(EllipticExample):
    def __init__(self, omega=5, alpha=3.0):
        self.omega = omega
        self.alpha = alpha
        super().__init__(x0=0.0, x1=1.0)

    def solution(self, x):
        w = self.omega * np.pi
        a = self.alpha
        return np.sin(w*x) + x*(1-x)*np.exp(a*x)

    def dsolution(self, x):
        w = self.omega * np.pi
        a = self.alpha
        return (
            w*np.cos(w*x)
            + np.exp(a*x) * (1 - 2*x + a*x*(1-x))
        )

    def f_vec(self, x, u):
        w = self.omega * np.pi
        a = self.alpha

        # g=x(1-x)e^{a x}
        # g''=e^{a x}[-2 + 2a(1-2x) + a^2 x(1-x)]
        return (
            w**2 * np.sin(w*x)
            - np.exp(a*x) * (
                -2 + 2*a*(1 - 2*x) + a**2*x*(1-x)
            )
        )