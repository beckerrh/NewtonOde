from .base_newton_driver import BaseNewtonDriver

#=================================================================#
class NewtonDriverOneLevel(BaseNewtonDriver):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def evaluate(self, x):
        from types import SimpleNamespace
        import numpy as np

        disc = self.model.discs[-1]

        F = disc.computeForm(x)
        b = disc.computeRhs()
        r = F.flatten() - b.flatten()

        if disc.dirichletmethod == "strong":
            bd = disc.boundary_dofs_global()
            r[bd] = 0.0


        resn = np.linalg.norm(r)
        merit = resn

        # print(f"evluate: {resn=}")

        return SimpleNamespace(
            residual=r,
            meritvalue=merit,
            norm_X=np.linalg.norm(x.flatten()),
        )

    def compute_newton_step(self, x, state, info):
        from types import SimpleNamespace
        import numpy as np

        disc = self.model.discs[-1]

        A = disc.computeMatrix(x)
        r = state.residual

        self.model.B.update(A=A)

        p_flat = self.model.B.solve(
            b=-r,
            x0=np.zeros_like(r),
        )

        p = x.from_flat_like(p_flat)



        return SimpleNamespace(
            dx=p,
            dx_norm=np.linalg.norm(p_flat),
            liniter=getattr(self.model.B, "niter", 0),
            x=x,
            success=True,
        )

