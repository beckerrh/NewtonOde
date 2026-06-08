# FEM2D/models/operators/convection_operator.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from Utility.analyticalfunction import AnalyticalFunction

from .operator import Operator, xyz_from_points
from ...fems import rt0
from ...fems import data as femdata


# ================================================================= #
@dataclass(frozen=True, kw_only=True)
class ConvectionOperator(Operator):
    """
    Linear convection operator beta · grad u.

    Canonical representation:
        convdata.betart    RT0 face fluxes
        convdata.betacell  cellwise beta vectors

    method:
        "centered", "supg", or "lps"
    """

    convdata: object
    method: str = "lps"
    lpsparam: float = 0.2
    convection_fct: object | None = None

    def __repr__(self):
        if self.method == "lps":
            return (
                f"convection={self.method}\n"
                f"lpsparam={self.lpsparam}"
            )
        return f"convection={self.method}"

    # ------------------------------------------------------------
    # construction
    # ------------------------------------------------------------
    @classmethod
    def from_problemdata(
            cls,
            problemdata,
            mesh,
            fem,
            ncomp: int,
            method: str = "lps",
            lpsparam: float = 0.2,
    ):
        found, value = problemdata.params.lookup(
            "convection",
            ("data", "fct_glob"),
        )

        if found is None:
            return None

        convdata, convection_fct = cls._build_convection_data(
            value=value,
            source=found,
            mesh=mesh,
            fem=fem,
        )

        return cls(
            convdata=convdata,
            method=method,
            lpsparam=lpsparam,
            convection_fct=convection_fct,
            ncomp=ncomp
        )

    @staticmethod
    def _build_convection_data(value, source, mesh, fem):
        convdata = femdata.ConvectionData()
        rt = rt0.RT0(mesh=mesh)

        if source == "fct_glob":
            convection_fct = ConvectionOperator._parse_convection_function(
                value,
                dim=mesh.dimension,
            )
            convdata.betart = rt.interpolate(convection_fct)

        elif source == "data":
            data, source_fem, stack_storage = value
            convection_fct = None
            convdata.betart = rt.interpolateFromFem(
                data,
                source_fem,
                stack_storage,
            )

        else:
            raise ValueError(f"unknown convection source {source!r}")

        convdata.betacell = rt.toCell(convdata.betart)

        return convdata, convection_fct

    @staticmethod
    def _parse_convection_function(value, dim):
        if not isinstance(value, list):
            raise ValueError(
                "problemdata.params.fct_glob['convection'] must be "
                "a list of length dim of strings or AnalyticalFunction"
            )

        if len(value) != dim:
            raise ValueError(
                f"convection dimension mismatch: {dim=} but len(convection)={len(value)}"
            )

        if isinstance(value[0], str):
            return [AnalyticalFunction(expr=e) for e in value]

        if not isinstance(value[0], AnalyticalFunction):
            raise ValueError(
                "convection should be given as strings or AnalyticalFunction, "
                f"not {type(value[0])}"
            )

        return value

    # ------------------------------------------------------------
    # diagnostics / boundary
    # ------------------------------------------------------------
    def inflow_colors(self, mesh):
        colors = []

        for color, faces in mesh.labels.boundary.items():
            if np.any(self.convdata.betart[faces] < -1e-10):
                colors.append(color)

        return colors

    def check_inflow_is_dirichlet(self, mesh, bdrycond):
        colorsinflow = self.inflow_colors(mesh)
        colorsdir = bdrycond.colorsOfType("Dirichlet")

        if not set(colorsinflow).issubset(set(colorsdir)):
            raise ValueError(
                "Inflow boundaries need to be subset of Dirichlet boundaries "
                f"{colorsinflow=} {colorsdir=}"
            )

    # ------------------------------------------------------------
    # form
    # ------------------------------------------------------------
    def add_form(self, disc, DU, U):
        ncomp = U.shape[0]

        for icomp in range(ncomp):
            dui = DU[icomp]
            ui = U[icomp]

            if self.method == "centered":
                disc.fem.computeFormTransportCellWise(
                    dui,
                    ui,
                    self.convdata,
                    type="centered",
                )

            elif self.method == "supg":
                disc.fem.computeFormTransportCellWise(
                    dui,
                    ui,
                    self.convdata,
                    type="supg",
                )

            elif self.method == "lps":
                disc.fem.computeFormTransportCellWise(
                    dui,
                    ui,
                    self.convdata,
                    type="centered",
                )
                disc.fem.computeFormLps(
                    dui,
                    ui,
                    self.convdata.betart,
                    lpsparam=self.lpsparam,
                )

            else:
                raise ValueError(f"unknown convection method {self.method!r}")

    # ------------------------------------------------------------
    # matrix
    # ------------------------------------------------------------
    def add_matrix(self, disc, A, u=None):
        fem = disc.fem
        C0 = self.scalar_matrix(fem)

        for i in range(self.ncomp):
            A[i][i] += C0

        return A

    def scalar_matrix(self, fem):
        if self.method == "centered":
            return fem.computeMatrixTransportCellWise(
                self.convdata,
                type="centered",
            )

        if self.method == "supg":
            return fem.computeMatrixTransportCellWise(
                self.convdata,
                type="supg",
            )

        if self.method == "lps":
            C = fem.computeMatrixTransportCellWise(
                self.convdata,
                type="centered",
            )
            C += fem.computeMatrixLps(
                self.convdata.betart,
                lpsparam=self.lpsparam,
            )
            return C

        raise ValueError(f"unknown convection method {self.method!r}")

    # ------------------------------------------------------------
    # RHS / estimator support
    # ------------------------------------------------------------
    def add_dirichlet_inflow_rhs(
        self,
        disc,
        B,
        colorsdir,
        bdrycondfct_by_comp,
    ):
        """
        Adds the inflow boundary term for imposed Dirichlet data:

            ∫_{Γ_D} g v (-min(beta·n, 0)).
        """
        ncomp = B.shape[0]

        for icomp in range(ncomp):
            fp1_i = disc.fem.interpolateBoundary(
                colorsdir,
                bdrycondfct_by_comp[icomp],
            )

            disc.fem.massDotBoundary(
                B[icomp],
                fp1_i,
                colors=colorsdir,
                coeff=-np.minimum(self.convdata.betart, 0),
            )

    def subtract_from_cell_rhs(self, disc, rhs_cell, U):
        """
        For residual estimator:

            f - beta · grad u - r(u)
        """
        ncomp = U.shape[0]
        beta = self.convdata.betacell[:, : disc.mesh.dimension]

        for i in range(ncomp):
            grad_i = disc.fem.cell_grad(U[i])
            rhs_cell[i] -= np.einsum("nd,nd->n", beta, grad_i)

        return rhs_cell

    def add_to_manufactured_rhs(self, out, solexact, x, y, z):
        if self.convection_fct is None:
            raise NotImplementedError(
                "Manufactured RHS with convection requires analytical convection_fct"
            )

        dim = len(self.convection_fct)

        for i, ui in enumerate(solexact):
            for d in range(dim):
                out[i] += self.convection_fct[d](x, y, z) * ui.d(d, x, y, z)