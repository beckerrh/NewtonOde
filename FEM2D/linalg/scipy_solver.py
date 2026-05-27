import scipy.sparse.linalg as splinalg

from FEM2D.linalg.solver import IterativeSolver


scipysolvers = [
    "scipy_gmres",
    "scipy_lgmres",
    "scipy_gcrotmk",
    "scipy_bicgstab",
    "scipy_cgs",
    "scipy_cg",
]

pyamgsolvers = [
    "pyamg_fgmres",
    "pyamg_bicgstab",
    "pyamg_cg",
]

othersolvers = ["idr"]

#=================================================================#
class ScipySpSolve:
    def __init__(self, **kwargs):
        self.matrix = kwargs.pop("matrix", None)
        self.method = "spsolve"
        self.niter = 1

    def solve(self, A=None, b=None, **kwargs):
        if A is None:
            A = self.matrix

        if hasattr(A, "to_single_matrix"):
            A = A.to_single_matrix()

        return splinalg.spsolve(A, b)

    def update(self, A):
        self.matrix = A
#=================================================================#
class ScipySolve(IterativeSolver):
    def __init__(self, **kwargs):
        self.method = kwargs.pop("method")
        self.args = {}

        self.n = kwargs.pop("n", None)
        matrix = kwargs.pop("matrix", None)
        matvec = kwargs.pop("matvec", None)
        preconditioner = kwargs.pop("preconditioner", None)
        matvecprec = kwargs.pop("matvecprec", None)

        if preconditioner is not None:
            if self.n is None:
                if matrix is not None:
                    self.n = matrix.shape[0]
                else:
                    raise ValueError("need 'n' or 'matrix' with preconditioner")
            self.preconditioner = preconditioner
            self.M = splinalg.LinearOperator(
                shape=(self.n, self.n),
                matvec=self.preconditioner.solve,
            )

        if matrix is not None:
            self.matvec = matrix

            if matvecprec is not None:
                self.M = splinalg.LinearOperator(
                    matrix.shape,
                    matvec=matvecprec,
                )
            elif not hasattr(self, "M"):
                fill_factor = kwargs.pop("fill_factor", 2)
                drop_tol = kwargs.pop("drop_tol", 0.01)
                spilu = splinalg.spilu(
                    self.matvec.tocsc(),
                    drop_tol=drop_tol,
                    fill_factor=fill_factor,
                )
                self.M = splinalg.LinearOperator(
                    self.matvec.shape,
                    lambda x: spilu.solve(x),
                )

        else:
            if self.n is None:
                raise ValueError("need 'n' if no matrix given")
            if matvec is None:
                raise ValueError("need 'matvec' if no matrix given")

            self.matvec = splinalg.LinearOperator(
                shape=(self.n, self.n),
                matvec=matvec,
            )

        self.args["callback_type"] = kwargs.pop("callback_type", "pr_norm")
        self.args["A"] = self.matvec
        self.args["M"] = getattr(self, "M", None)

        if self.method == "scipy_gcrotmk":
            self.args["m"] = kwargs.pop("m", 5)
            self.args["truncate"] = kwargs.pop("truncate", "smallest")
            self.solver = splinalg.gcrotmk

        elif self.method == "scipy_lgmres":
            self.args["inner_m"] = kwargs.pop("m", 20)
            self.solver = splinalg.lgmres

        elif self.method in scipysolvers:
            self.solver = getattr(splinalg, self.method[6:])

        elif self.method in pyamgsolvers:
            import pyamg
            self.solver = getattr(pyamg.krylov, self.method[6:])

        elif self.method == "idr":
            import idrs
            self.solver = idrs.idrs

        else:
            raise ValueError(f"unknown {self.method=}")

        super().__init__(kwargs)

        if kwargs:
            raise ValueError(f"*** unused arguments {kwargs=}")

    def _solve_impl(self, A=None, b=None, x0=None, maxiter=None, rtol=None, atol=None):
        args = dict(self.args)

        if A is not None:
            args["A"] = A

        args["b"] = b
        args["x0"] = x0
        args["maxiter"] = maxiter
        args["callback"] = self.callback

        if self.method in scipysolvers:
            args["rtol"] = rtol
            args["atol"] = atol
        else:
            args["tol"] = rtol

        res = self.solver(**args)
        return res[0] if isinstance(res, tuple) else res

    def update(self, A=None, **kwargs):
        if A is not None:
            self.matvec = A
            self.args["A"] = A
        if hasattr(self, "preconditioner"):
            self.preconditioner.update(A)
            self.M = splinalg.LinearOperator(
                shape=A.shape,
                matvec=self.preconditioner.solve,
            )
            self.args["M"] = self.M