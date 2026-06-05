import numpy as np

from .model import Model
from .elliptic_discretization import EllipticDiscretization


# ================================================================= #
class Elliptic(Model):
    def __init__(self, **kwargs):
        self.fem_name = kwargs.pop("fem", "cr1")
        super().__init__(**kwargs)

    # ================================================================= #
    def discretize(self, mesh):
        return EllipticDiscretization(
            mesh=mesh,
            application=self.application,
            fem_name=self.fem_name,
            disc_params=self.disc_params,
            problemdata=self.problemdata,
        )

    # ================================================================= #
    def setParameter(self, paramname, param):
        if paramname == "dirichlet_strong":
            self.fem.dirichlet_strong = param
            return

        if not hasattr(self, paramname):
            raise NotImplementedError(
                f"{type(self).__name__} has no parameter {paramname!r}"
            )

        setattr(self, paramname, param)