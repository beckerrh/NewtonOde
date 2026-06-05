class NewtonDriver:
    def __init__(self, model):
        self.model = model

    def initial_guess(self):
        disc = self.model.discs[-1]
        return disc.newVector()

    def add_update(self, x, alpha, p):
        return x.from_flat_like(x.flatten() + alpha * p.flatten())

    def evaluate(self, x):
        from types import SimpleNamespace
        import numpy as np

        disc = self.model.discs[-1]

        F = disc.computeForm(x)
        b = disc.computeRhs()
        r = F.flatten() - b.flatten()

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

        A = disc.computeMatrix()
        r = state.residual

        self.model.B.update(A=A)

        p_flat = self.model.B.solve(
            b=-r,
            x0=np.zeros_like(r),
        )

        p = x.from_flat_like(p_flat)

        # A = disc.computeMatrix()
        # b = disc.computeRhs()
        # r_matrix = A @ x.flatten() - b.flatten()
        # r_form = disc.computeForm(x).flatten() - b.flatten()
        #
        # print("matrix residual", np.linalg.norm(r_matrix))
        # print("form residual  ", np.linalg.norm(r_form))
        # print("form-matrix diff", np.linalg.norm(r_form - r_matrix))



        return SimpleNamespace(
            dx=p,
            dx_norm=np.linalg.norm(p_flat),
            meritgrad=-state.meritvalue,
            liniter=getattr(self.model.B, "niter", 0),
            x=x,
            success=True,
        )