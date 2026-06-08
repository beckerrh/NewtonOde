#=================================================================#

class BaseNewtonDriver:
    def __init__(self, model, **kwargs):
        self.model = model
        self.verbose= kwargs.pop('verbose', True)
        self.debug = kwargs.pop('debug', False)

    def attach_logger(self, logger):
        self.logger_newton = logger
        types = {'eta':'e', 'N': 'd:7'}
        # types = {'eta':'e', '|F_h|':'e', 'N': 'd:7', 'meshiter': 'd:3'}
        self.logger_newton.add_types(types)
        self.logger_newton.update(N=self.model.discs[-1].mesh.ncells)

    def initial_guess(self):
        disc = self.model.discs[-1]
        return disc.initial_guess()

    def add_update(self, x, alpha, p):
         return x + alpha * p

    def check_jacobian_fd(self, x):
        import numpy as np

        disc = self.model.discs[-1]
        rng = np.random.default_rng(0)

        p = x.zeros_like()
        p.part("U")[:] = rng.standard_normal(p.part("U").shape)

        p_flat = p.flatten()

        # For strong Dirichlet, perturb only admissible/free DOFs.
        if getattr(disc, "dirichletmethod", None) == "strong":
            bd = disc.boundary_dofs_global()
            p_flat[bd] = 0.0
            p = x.from_flat_like(p_flat)
        else:
            bd = np.array([], dtype=int)

        F0 = disc.computeForm(x).flatten() - disc.computeRhs().flatten()
        A = disc.computeMatrix(x)

        eps = 1e-7
        x1 = x.from_flat_like(x.flatten() + eps * p_flat)
        F1 = disc.computeForm(x1).flatten() - disc.computeRhs().flatten()

        fd = (F1 - F0) / eps
        Ap = A @ p_flat
        err = fd - Ap

        print("J check abs", np.linalg.norm(err))
        print("J check rel", np.linalg.norm(err) / max(1.0, np.linalg.norm(fd)))
        print("fd norm", np.linalg.norm(fd), "Ap norm", np.linalg.norm(Ap))

        if bd.size:
            free = np.ones(A.shape[0], dtype=bool)
            free[bd] = False

            print(
                "J free rel",
                np.linalg.norm(err[free]) / max(1.0, np.linalg.norm(Ap[free])),
            )
            print(
                "J bd rel",
                np.linalg.norm(err[bd]) / max(1.0, np.linalg.norm(Ap[bd])),
            )
