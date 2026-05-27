from FEM2D.linalg.scipy_solver import (
    ScipySolve,
    ScipySpSolve,
    scipysolvers,
    pyamgsolvers,
    othersolvers,
)
from FEM2D.linalg.pyamg_solver import Pyamg
from FEM2D.linalg.multigrid import GeometricMultigrid
from FEM2D.linalg.scipy_solver import ScipySolve
from FEM2D.linalg.scipy_solver import scipysolvers, pyamgsolvers, othersolvers


#-------------------------------------------------------------------#
def getLinearSolver(**kwargs):
    method = kwargs.pop("method", "pyamg")
    matrix = kwargs.pop("matrix", None)

    As = kwargs.pop("As", None)
    transfers = kwargs.pop("transfers", None)

    if method == "geommg":
        if As is None or transfers is None:
            raise ValueError("geommg needs As and transfers")
        return GeometricMultigrid(As=As, transfers=transfers, **kwargs)

    if method in scipysolvers or method in pyamgsolvers or method in othersolvers:
        return ScipySolve(matrix=matrix, method=method, **kwargs)

    if method == "spsolve":
        return ScipySpSolve(matrix=matrix, **kwargs)

    if method == "pyamg":
        return Pyamg(matrix, **kwargs)

    raise ValueError(f"unknown {method=}")