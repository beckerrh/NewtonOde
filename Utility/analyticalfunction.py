# -*- coding: utf-8 -*-
"""
Small SymPy-based helper for manufactured analytical solutions.

Intended for tests/examples only, not for performance-critical FEM assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sp


@dataclass
class AnalyticalFunction:
    expr: str
    dim: int = 2
    has_time: bool = False

    def _trim_args(self, args):
        n = self.dim + int(self.has_time)
        return args[:n]

    def __post_init__(self) -> None:
        if self.dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2, or 3, got {self.dim}")

        expr = self._normalize_expr(self.expr, self.dim)
        self.expr = expr
        self.has_time = "t" in expr

        varnames = [f"x{i}" for i in range(self.dim)]
        if self.has_time:
            varnames.append("t")

        self.symbols = sp.symbols(" ".join(varnames))
        if self.dim == 1 and not self.has_time:
            self.symbols = (self.symbols,)

        self.sym_expr = sp.sympify(expr)

        self.fct = self._lambdify(self.sym_expr)

        self.fct_x: list[Callable] = []
        self.fct_xx: list[list[Callable]] = []

        spatial_symbols = self.symbols[: self.dim]

        for i, si in enumerate(spatial_symbols):
            d_i = sp.diff(self.sym_expr, si)
            self.fct_x.append(self._lambdify(d_i))

            row = []
            for sj in spatial_symbols:
                d_ij = sp.diff(d_i, sj)
                row.append(self._lambdify(d_ij))
            self.fct_xx.append(row)

        if self.has_time:
            self.fct_t = self._lambdify(sp.diff(self.sym_expr, self.symbols[-1]))

    def __repr__(self) -> str:
        return f"AnalyticalFunction(dim={self.dim}, expr='{self.expr}')"


    def __call__(self, *x):
        return self.fct(*self._trim_args(x))

    def _lambdify(self, expr) -> Callable:
        return sp.lambdify(self.symbols, expr, modules="numpy")

    @staticmethod
    def _normalize_expr(expr: str, dim: int) -> str:
        """
        Accept both x,y,z and x0,x1,x2 notation.

        For dim=2:
            'sin(pi*x)*sin(pi*y)' -> 'sin(pi*x0)*sin(pi*x1)'
        """
        if any(f"x{i}" in expr for i in range(3)):
            return expr

        replacements = [("x", "x0"), ("y", "x1"), ("z", "x2")]
        for old, new in replacements[:dim]:
            expr = expr.replace(old, new)
        return expr

    def d(self, i: int, *x):
        return self.fct_x[i](*self._trim_args(x))

    def dd(self, i: int, j: int, *x):
        return self.fct_xx[i][j](*self._trim_args(x))

    def t(self, *x):
        if not self.has_time:
            raise AttributeError("This analytical function has no time variable.")
        return self.fct_t(*self._trim_args(x))

    # Convenience aliases
    def x(self, *args):
        return self.d(0, *args)

    def y(self, *args):
        return self.d(1, *args)

    def z(self, *args):
        return self.d(2, *args)

    def xx(self, *args):
        return self.dd(0, 0, *args)

    def xy(self, *args):
        return self.dd(0, 1, *args)

    def yx(self, *args):
        return self.dd(1, 0, *args)

    def yy(self, *args):
        return self.dd(1, 1, *args)

    def laplace(self, *x):
        val = 0.0
        for i in range(self.dim):
            val = val + self.dd(i, i, *x)
        return val


def analytical_solution(function: str, dim: int = 2, ncomp: int = 1, random: bool = False):
    """
    Build simple manufactured scalar/vector analytical functions.

    Parameters
    ----------
    function:
        Either one of {'Constant', 'Linear', 'Quadratic', 'Sinus'}
        or an explicit SymPy-compatible expression.
    dim:
        Spatial dimension.
    ncomp:
        Number of components.
    random:
        Whether to use random coefficients for generated functions.
    """
    if dim not in (1, 2, 3):
        raise ValueError(f"dim must be 1, 2, or 3, got {dim}")

    rng = np.random.default_rng(12345)

    def coeffs(n: int):
        if random:
            return list((4.0 * rng.random(n) - 2.0) / 3.0)
        return [1.1 * (n - k) for k in range(n)]

    varnames = ["x", "y", "z"][:dim]
    p = coeffs(max(1, ncomp * (1 + 2 * dim)))

    functions = []

    for comp in range(ncomp):
        if function == "Constant":
            expr = f"{p.pop():.8g}"

        elif function in ("Linear", "Quadratic"):
            expr = f"{p.pop():.8g}"
            for var in varnames:
                expr += f"{p.pop():+.8g}*{var}"

            if function == "Quadratic":
                for var in varnames:
                    expr += f"{p.pop():+.8g}*{var}**2"

        elif function == "Sinus":
            expr = f"{p.pop():.8g}"
            for var in varnames:
                expr += f"{p.pop():+.8g}*sin({var})"

        else:
            if ncomp == 1:
                expr = function
            else:
                expr = function[comp]

        functions.append(AnalyticalFunction(expr=expr, dim=dim))

    return functions[0] if ncomp == 1 else functions


if __name__ == "__main__":
    u = AnalyticalFunction("sin(pi*x)*sin(pi*y)", dim=2)

    x = np.linspace(0.0, 1.0, 4)
    y = np.linspace(0.0, 1.0, 4)
    X, Y = np.meshgrid(x, y, indexing="ij")

    print(u)
    print("u =", u(X, Y))
    print("ux =", u.x(X, Y))
    print("uy =", u.y(X, Y))
    print("laplace u =", u.laplace(X, Y))