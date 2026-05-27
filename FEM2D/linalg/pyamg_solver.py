from FEM2D.linalg.solver import IterativeSolver


#=================================================================#
class Pyamg(IterativeSolver):
    def __repr__(self):
        s = super().__repr__()
        return s + f"_pyamg_{self.type}_{self.smoother}_{str(self.accel)}"

    def __init__(self, A, **kwargs):
        try:
            import pyamg
        except ImportError:
            raise ImportError("*** pyamg not found ***")


        self.method = "pyamg"

        nsmooth = kwargs.pop("nsmooth", 1)
        symmetric = kwargs.pop("symmetric", False)
        self.smoother = kwargs.pop("smoother", "gauss_seidel")
        self.type = kwargs.pop("pyamgtype", "aggregation")

        if symmetric:
            default_accel = "cg"
            smooth = ("energy", {"krylov": "cg"})
            pyamgargs = {}
        else:
            default_accel = "fgmres"
            smooth = ("energy", {"krylov": "fgmres"})
            pyamgargs = {"symmetry": "nonsymmetric"}

        smoother = (self.smoother, {"sweep": "symmetric", "iterations": nsmooth})

        pyamgargs["presmoother"] = smoother
        pyamgargs["postsmoother"] = smoother
        pyamgargs["smooth"] = smooth
        pyamgargs["coarse_solver"] = "splu"

        self.pyamgargs = pyamgargs
        self.accel = kwargs.pop("accel", default_accel)
        self.cycle = kwargs.pop("cycle", "V")

        if A is not None:
            self._build_solver(A)

        super().__init__(kwargs)

        if kwargs:
            raise ValueError(f"*** unused arguments {kwargs=}")

    def _build_solver(self, A):
        import pyamg

        if self.type == "aggregation":
            self.mlsolver = pyamg.smoothed_aggregation_solver(A, **self.pyamgargs)
        elif self.type == "rootnode":
            self.mlsolver = pyamg.rootnode_solver(A, **self.pyamgargs)
        else:
            raise ValueError(f"unknown {self.type=}")

    def update(self, A):
        self._build_solver(A)

    def _solve_impl(self, A=None, b=None, x0=None, maxiter=None, rtol=None, atol=None):
        if A is not None:
            self.update(A)

        return self.mlsolver.solve(
            b=b,
            x0=x0,
            maxiter=maxiter,
            tol=rtol,
            accel=self.accel,
            cycle=self.cycle,
            callback=self.callback,
        )