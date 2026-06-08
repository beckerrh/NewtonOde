import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


# ================================================================= #
@dataclass(frozen=True, kw_only=True)
class Operator(ABC):
    ncomp: int

    def __post_init__(self):
        if self.ncomp <= 0:
            raise ValueError(f"{self.ncomp=}")

    def _as_component_cell(self, X, ncells):
        X = np.asarray(X)

        if X.ndim == 0:
            return np.full((self.ncomp, ncells), float(X))

        if X.shape == (ncells,):
            return np.repeat(X[None, :], self.ncomp, axis=0)

        if X.shape == (self.ncomp, ncells):
            return X

        raise ValueError(
            f"bad shape {X.shape=} "
            f"expected ({ncells},) or ({self.ncomp}, {ncells})"
        )

    def _as_component_dof(self, X, ndofs):
        X = np.asarray(X)

        if X.ndim == 0:
            return np.full((self.ncomp, ndofs), float(X))

        if X.shape == (ndofs,):
            return np.repeat(X[None, :], self.ncomp, axis=0)

        if X.shape == (self.ncomp, ndofs):
            return X

        raise ValueError(
            f"bad component-dof shape {X.shape=}, "
            f"expected (), ({ndofs},), or ({self.ncomp}, {ndofs})"
        )

    def _to_cell_block(self, disc, U):
        return np.vstack([
            disc.fem.to_cell(U[i])
            for i in range(self.ncomp)
        ])

    @abstractmethod
    def add_form(self, disc, DU, U):
        raise NotImplementedError

    @abstractmethod
    def add_matrix(self, disc, A, u=None):
        raise NotImplementedError


# ================================================================= #
def xyz_from_points(points):
    points = np.asarray(points)

    if points.shape[1] == 2:
        x, y = points.T
        z = np.zeros_like(x)
        return x, y, z

    if points.shape[1] == 3:
        return points.T

    raise ValueError(f"expected points with dim 2 or 3, got {points.shape=}")
