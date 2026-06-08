# FEM2D/models/operators/diffusion_operator.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import block_diag, bmat

from .operator import Operator, xyz_from_points
from .normalize_diffusion import normalize_diffusion


# ================================================================= #
@dataclass(frozen=True, kw_only=True)
class DiffusionOperator(Operator):
    """
    Linear diffusion operator.

    Canonical internal representation:
        kind == "scalar"   : diffcell.shape == (ncells,)
        kind == "diagonal" : diffcell.shape == (ncomp, ncells)
        kind == "coupled"  : diffcell.shape == (ncomp, ncomp, ncells)

    The weak contribution is

        sum_j ∫ k_ij grad u_j · grad v_i.
    """

    kind: str
    diffcell: np.ndarray

    def __repr__(self):
        return f"diffusion={self.kind}"
    # ------------------------------------------------------------
    # construction / normalization
    # ------------------------------------------------------------
    @classmethod
    def from_problemdata(cls, problemdata, mesh, fem, ncomp: int):
        kheatcell = cls._cell_vector_from_params(
            "kheat",
            problemdata.params,
            mesh,
        )

        kind, diffcell = normalize_diffusion(
            kheatcell,
            mesh.ncells,
            ncomp=ncomp,
            dim=mesh.dimension,
        )

        return cls(kind=kind, diffcell=diffcell, ncomp=ncomp)

    @staticmethod
    def _cell_vector_from_params(name, params, mesh):
        kind, value = params.lookup(
            name,
            ("fct_glob", "scal_glob", "scal_cells", "scal_celllabels"),
        )

        if kind == "fct_glob":
            fct = np.vectorize(value)
            arr = np.empty(mesh.ncells)

            for color, cells in mesh.labels.cell.items():
                xc, yc, zc = xyz_from_points(mesh.geometry.cell_centers[cells])
                arr[cells] = fct(color, xc, yc, zc)

            return arr

        if kind == "scal_glob":
            return np.full(mesh.ncells, value)

        if kind == "scal_celllabels":
            arr = np.empty(mesh.ncells)

            for color, val in value.items():
                arr[mesh.labels.cell[color]] = val

            return arr

        if kind == "scal_cells":
            arr = np.asarray(value)
            if arr.shape != (mesh.ncells,):
                raise ValueError(
                    f"{name} in scal_cells has shape {arr.shape}, "
                    f"expected {(mesh.ncells,)}"
                )
            return arr

        raise ValueError(
            f"{name} should be given in params.fct_glob, "
            f"params.scal_glob, or params.scal_celllabels"
        )

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def component_coeff(self, icomp: int):
        if self.kind == "scalar":
            return self.diffcell

        if self.kind == "diagonal":
            return self.diffcell[icomp]

        raise ValueError(
            "component_coeff is only defined for scalar/diagonal diffusion"
        )

    def coeff_pointwise(self, i):
        if self.kind == "scalar":
            r = np.asarray(self.diffcell)
            if np.allclose(r, r.flat[0]):
                return float(r.flat[0])
            raise NotImplementedError("cellwise diffusion is not pointwise constant")

        if self.kind == "diagonal":
            r = np.asarray(self.diffcell)
            ri = r[i]
            if np.allclose(ri, ri.flat[0]):
                return float(ri.flat[0])
            raise NotImplementedError("cellwise diagonal diffusion is not pointwise constant")

        raise NotImplementedError("manufactured RHS for coupled diffusion")

    # ------------------------------------------------------------
    # form
    # ------------------------------------------------------------
    def add_form(self, disc, DU, U):
        ncomp = U.shape[0]

        if self.kind in ("scalar", "diagonal"):
            for icomp in range(ncomp):
                disc.fem.computeFormDiffusion(
                    DU[icomp],
                    U[icomp],
                    self.component_coeff(icomp),
                )
            return

        if self.kind == "coupled":
            for i in range(ncomp):
                for j in range(ncomp):
                    kij = self.diffcell[i, j]

                    if np.all(kij == 0):
                        continue

                    disc.fem.computeFormDiffusion(
                        DU[i],
                        U[j],
                        kij,
                    )
            return

        raise ValueError(f"unknown diffusion kind {self.kind!r}")

    # ------------------------------------------------------------
    # matrix
    # ------------------------------------------------------------
    def add_matrix(self, disc, A, U=None):
        fem = disc.fem
        ncomp = self.ncomp


        if self.kind in ("scalar", "diagonal"):
            for i in range(ncomp):
                A[i][i] += fem.computeMatrixDiffusion(
                    self.component_coeff(i)
                )
            return A

        if self.kind == "coupled":
            for i in range(ncomp):
                for j in range(ncomp):
                    kij = self.diffcell[i, j]
                    if np.all(kij == 0):
                        continue
                    A[i][j] += fem.computeMatrixDiffusion(kij)
            return A

        raise ValueError(f"unknown diffusion kind {self.kind!r}")
    # ------------------------------------------------------------
    # estimator support / plotting support
    # ------------------------------------------------------------
    def estimator_coeff(self, icomp: int):
        return self.component_coeff(icomp)

    def plot_cell_data(self):
        if self.kind == "scalar":
            return {"k": self.diffcell}

        if self.kind == "diagonal":
            return {
                f"k[{i}]": self.diffcell[i]
                for i in range(self.diffcell.shape[0])
            }

        if self.kind == "coupled":
            data = {}
            ncomp = self.diffcell.shape[0]
            for i in range(ncomp):
                for j in range(ncomp):
                    data[f"k[{i},{j}]"] = self.diffcell[i, j]
            return data

        return {}