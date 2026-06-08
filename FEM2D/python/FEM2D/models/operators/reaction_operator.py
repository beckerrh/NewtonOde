# FEM2D/models/operators/reaction_operator.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse import block_diag, bmat

from .operator import Operator, xyz_from_points
from .normalize_reaction import normalize_reaction


def _linear_reaction_from_coeff(coeff):
    coeff = np.asarray(coeff)

    def r(u):
        return coeff * u

    def rd(u):
        return coeff + 0 * u

    return r, rd


def _linear_reaction_from_scalar(c):
    c = float(c)

    def r(u):
        return c * u

    def rd(u):
        return c + 0 * u

    return r, rd

# ================================================================= #
@dataclass(frozen=True, kw_only=True)
class ReactionOperator(Operator):
    """
    Reaction operator r(u).

    Current version:
        linear reaction r_i(u) = sum_j c_ij u_j

    Canonical forms:
        kind == "scalar"   : reactioncell.shape == (ncells,)
        kind == "diagonal" : reactioncell.shape == (ncomp, ncells)
        kind == "coupled"  : reactioncell.shape == (ncomp, ncomp, ncells)

    Later nonlinear reactions should keep the same public interface:
        value_cell(Ucell)
        derivative_cell(Ucell)
    """

    kind: str
    reactioncell: object = None
    raw_reaction: object = None
    raw_reaction_d: object = None
    ncomp: int = 1
    lumped: bool = True
    fct: object = None
    fct_d: object = None

    def __repr__(self):
        return f"reaction={self.kind} lumped={self.lumped}"

    # ------------------------------------------------------------
    # construction
    # ------------------------------------------------------------
    @classmethod
    def from_problemdata(cls, problemdata, mesh, fem, ncomp: int, lumped: bool = True):
        reaction = cls._read_reaction(problemdata, mesh)

        if reaction is None:
            return None

        if np.isscalar(reaction):
            kind = "scalar"
            reactioncell = None
            raw_reaction, raw_reaction_d = _linear_reaction_from_scalar(reaction)

        elif isinstance(reaction, np.ndarray):
            kind, reactioncell = normalize_reaction(
                reaction,
                mesh.ncells,
                ncomp=ncomp,
            )
            raw_reaction, raw_reaction_d = _linear_reaction_from_coeff(reactioncell)

        elif isinstance(reaction, dict):
            kind = reaction.get("kind", "scalar")
            reactioncell = reaction.get("reactioncell", None)

            r = reaction.get("reaction", None)
            rd = reaction.get("reaction_d", None)

            if r is None:
                raise ValueError("reaction dict has no 'reaction' entry")

            # Linear coefficient encoded as dict
            if rd is None and not callable(r):
                kind, reactioncell = normalize_reaction(
                    r,
                    mesh.ncells,
                    ncomp=ncomp,
                )
                raw_reaction, raw_reaction_d = _linear_reaction_from_coeff(reactioncell)

            # Nonlinear reaction encoded as dict
            elif callable(r):
                if rd is None:
                    raise ValueError(
                        "nonlinear reaction dict needs callable 'reaction_d'"
                    )
                raw_reaction = r
                raw_reaction_d = rd


            else:
                raise ValueError(
                    f"invalid reaction dict: reaction={r}, reaction_d={rd}"
                    )

        else:
            raise TypeError(f"unknown reaction descriptor {type(reaction)=}")

        # print("REACTION DEBUG")
        # print("reaction =", reaction)
        # print("type(reaction) =", type(reaction))
        # print("raw_reaction =", raw_reaction)
        # print("raw_reaction_d =", raw_reaction_d)
        # print("kind =", kind)
        # print("reactioncell =", reactioncell)

        category, fct = problemdata.params.lookup("reaction", ["fct_glob"])
        category_d, fct_d = problemdata.params.lookup("reaction_d", ["fct_glob"])
        return cls(
            kind=kind,
            reactioncell=reactioncell,
            raw_reaction=raw_reaction,
            raw_reaction_d=raw_reaction_d,
            ncomp=ncomp,
            lumped=lumped,
            fct=fct,
            fct_d=fct_d,
        )

    @staticmethod
    def _read_reaction(problemdata, mesh):
        params = problemdata.params

        kind, reaction = params.lookup(
            "reaction",
            ("fct_glob", "scal_glob", "scal_cells", "scal_celllabels", "data"),
        )

        if kind is None:
            return None

        d_kind, reaction_d = params.lookup(
            "reaction_d",
            ("fct_glob", "scal_glob", "data"),
        )

        if kind == "scal_glob":
            arr = np.asarray(reaction, dtype=float)

            if arr.ndim == 0:
                return float(arr)

            if arr.ndim == 1:
                return {
                    "kind": "diagonal",
                    "reaction": arr,
                    "reaction_d": None,
                }

            if arr.ndim == 2:
                return {
                    "kind": "coupled",
                    "reaction": arr,
                    "reaction_d": None,
                }

            raise ValueError(f"bad reaction scal_glob shape {arr.shape}")

        elif kind == "scal_cells":
            reaction = np.asarray(reaction)
            if reaction.shape != (mesh.ncells,):
                raise ValueError(
                    f"reaction in scal_cells has shape {reaction.shape}, "
                    f"expected {(mesh.ncells,)}"
                )

        elif kind == "scal_celllabels":
            arr = np.empty(mesh.ncells)
            for color, value in reaction.items():
                arr[mesh.labels.cell[color]] = value
            reaction = arr

        elif kind == "fct_glob":
            # keep callable as callable
            pass

        elif kind == "data":
            # already user-provided reaction descriptor
            pass

        else:
            raise ValueError(f"unknown reaction source {kind!r}")

        return {
            "reaction": reaction,
            "reaction_d": reaction_d,
            "reaction_source": kind,
            "reaction_d_source": d_kind,
        }

    def _cell_vector_from_function(fct, mesh):
        fct = np.vectorize(fct)
        arr = np.empty(mesh.ncells)

        for color, cells in mesh.labels.cell.items():
            xc, yc, zc = xyz_from_points(mesh.geometry.cell_centers[cells])
            arr[cells] = fct(color, xc, yc, zc)

        return arr
    @staticmethod
    def _cell_vector_from_params(name, params, mesh):
        if name in params.fct_glob:
            fct = np.vectorize(params.fct_glob[name])
            arr = np.empty(mesh.ncells)

            for color, cells in mesh.labels.cell.items():
                xc, yc, zc = xyz_from_points(mesh.geometry.cell_centers[cells])
                arr[cells] = fct(color, xc, yc, zc)

        elif name in params.scal_glob:
            arr = np.full(mesh.ncells, params.scal_glob[name])

        elif name in params.scal_cells:
            arr = np.empty(mesh.ncells)

            for color, value in params.scal_cells[name].items():
                arr[mesh.labels.cell[color]] = value

        else:
            raise ValueError(
                f"{name} should be given in params.fct_glob, "
                f"params.scal_glob, or params.scal_cells"
            )

        return arr

   # ------------------------------------------------------------
    # coefficients
    # ------------------------------------------------------------
    def coeff(self, i: int, j: int):
        if self.kind == "scalar":
            return self.reactioncell if i == j else 0.0

        if self.kind == "diagonal":
            return self.reactioncell[i] if i == j else 0.0

        if self.kind == "coupled":
            return self.reactioncell[i, j]

        raise ValueError(f"unknown reaction kind {self.kind!r}")

    # ------------------------------------------------------------
    # nonlinear-compatible interface
    # ------------------------------------------------------------
    # def value_cell(self, Ucell):
    #     """
    #     Ucell.shape == (ncomp, ncells)
    #     returns Rcell.shape == (ncomp, ncells)
    #     """
    #     ncomp = Ucell.shape[0]
    #     Rcell = np.zeros_like(Ucell)
    #
    #     print(f"{self.fct=}")
    #
    #     for i in range(ncomp):
    #         for j in range(ncomp):
    #             cij = self.coeff(i, j)
    #
    #             if np.isscalar(cij) and cij == 0:
    #                 continue
    #
    #             Rcell[i] += cij * Ucell[j]
    #
    #     return Rcell
    def value_cell(self, Ucell):
        import numpy as np

        if self.fct is not None:
            return np.asarray(self.fct(Ucell))

        Rcell = np.zeros_like(Ucell)

        for i in range(self.ncomp):
            for j in range(self.ncomp):
                cij = self.coef[i][j]
                if cij is None:
                    continue
                Rcell[i] += cij * Ucell[j]

        return Rcell
    def derivative_cell(self, Ucell):
        """
        Linear reaction: derivative is independent of Ucell.

        returns either canonical reactioncell data.
        """
        return self.reactioncell

    # ------------------------------------------------------------
    # form
    # ------------------------------------------------------------
    def add_form(self, disc, DU, U):
        if self.lumped:
            Ucell = self._to_cell_block(disc, U)
            R = np.asarray(self.raw_reaction(Ucell))
            R = self._as_component_cell(R, disc.mesh.ncells)

            for i in range(self.ncomp):
                disc.fem.massDotCell(DU[i], R[i])

        else:
            R = np.asarray(self.raw_reaction(U))
            R = self._as_component_dof(R, disc.fem.nunknowns())

            for i in range(self.ncomp):
                disc.fem.massDot(DU[i], R[i])
    # ------------------------------------------------------------
    # matrix
    # ------------------------------------------------------------
    def add_matrix(self, disc, A, U):
        if self.lumped:
            Ucell = self._to_cell_block(disc, U)
            D = np.asarray(self.raw_reaction_d(Ucell))

            if self.kind in ("scalar", "diagonal"):
                D = self._as_component_cell(D, disc.mesh.ncells)

                for i in range(self.ncomp):
                    A[i][i] += disc.fem.computeMassMatrixCellReaction(D[i])
                    # A[i][i] += disc.fem.computeMassMatrixCellAverageReaction(D[i])
                    # A[i][i] += disc.fem.computeMassMatrix(
                    #     coeff=D[i],
                    #     lumped=True,
                    # )

                return A

            if self.kind == "coupled":
                D = np.asarray(D)

                if D.shape != (self.ncomp, self.ncomp, disc.mesh.ncells):
                    raise ValueError(f"{D.shape=}")

                for i in range(self.ncomp):
                    for j in range(self.ncomp):
                        A[i][j] += disc.fem.computeMassMatrix(
                            coeff=D[i, j],
                            lumped=True,
                        )

                return A

            raise ValueError(f"unknown reaction kind {self.kind!r}")

        # non-lumped: evaluate derivative at dofs
        # non-lumped: evaluate derivative at dofs
        D = np.asarray(self.raw_reaction_d(U))

        if self.kind in ("scalar", "diagonal"):
            D = self._as_component_dof(D, disc.fem.nunknowns())

            M = disc.fem.computeMassMatrix(coeff=1, lumped=False)

            for i in range(self.ncomp):
                A[i][i] += M @ sparse.diags(D[i])

            return A

        if self.kind == "coupled":
            D = np.asarray(D)

            if D.shape != (self.ncomp, self.ncomp, disc.fem.nunknowns()):
                raise ValueError(f"{D.shape=}")

            M = disc.fem.computeMassMatrix(coeff=1, lumped=False)

            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    A[i][j] += M @ sparse.diags(D[i, j])

            return A
        raise ValueError(f"unknown reaction kind {self.kind!r}")

    # ------------------------------------------------------------
    # estimator support
    # ------------------------------------------------------------
    def subtract_from_cell_rhs(self, disc, rhs_cell, U):
        ncomp = U.shape[0]

        Ucell = np.vstack([
            disc.fem.to_cell(U[j])
            for j in range(ncomp)
        ])

        rhs_cell -= self.value_cell(Ucell)

        return rhs_cell

    def coeff_pointwise(self, i, j):
        if self.reactioncell is None:
            raise ValueError("coeff_pointwise only works for linear coefficient reactions")

        r = np.asarray(self.reactioncell, dtype=float)

        if self.kind == "scalar":
            if r.size == 1:
                return float(r.ravel()[0])
            if np.allclose(r, r.ravel()[0]):
                return float(r.ravel()[0])
            raise ValueError("cellwise scalar reaction cannot be evaluated pointwise")

        if self.kind == "diagonal":
            if i != j:
                return 0.0

            ri = r[i] if r.ndim >= 2 else r

            if ri.size == 1:
                return float(ri.ravel()[0])
            if np.allclose(ri, ri.ravel()[0]):
                return float(ri.ravel()[0])

            raise ValueError("cellwise diagonal reaction cannot be evaluated pointwise")

        if self.kind in ("full", "coupled"):
            rij = r[i, j]
            if np.size(rij) == 1:
                return float(np.ravel(rij)[0])
            if np.allclose(rij, np.ravel(rij)[0]):
                return float(np.ravel(rij)[0])
            raise ValueError("cellwise coupled reaction has no pointwise constant value")


        raise ValueError(f"unknown reaction kind {self.kind=}")

    def add_to_manufactured_rhs(self, out, solexact, x, y, z):
        ncomp = len(solexact)

        for i in range(ncomp):
            for j in range(ncomp):
                cij = self.coeff_pointwise(i, j)
                if cij != 0:
                    out[i] += cij * solexact[j](x, y, z)

    def subtract_linearized_from_cell_rhs(self, disc, rhs_cell, U, W):
        Ucell = disc.cellmean_vector(U)
        Wcell = disc.cellmean_vector(W)

        fct_d = getattr(self, "fct_d", None)

        if fct_d is not None:
            rhs_cell -= fct_d(Ucell) * Wcell
            return rhs_cell

        rhs_cell -= self.value_cell(Wcell)
        return rhs_cell