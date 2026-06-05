import copy
import numpy as np
import pandas as pd


#==================================================================
class CompareMethods:
    def __init__(self, application, model, methods, nref=4, callback=None):
        self.application = application
        self.model = model
        self.methods = methods
        self.nref = nref
        self.callback = callback
        self.df = None

    def run(self):
        rows = []

        for name, method_params in self.methods.items():
            app = copy.deepcopy(self.application)
            modelargs = copy.deepcopy(method_params)
            modelargs["application"] = app

            m = self.model(**modelargs)

            for level in range(self.nref):
                disc = m.discs[-1]

                b = disc.computeRhs()
                A = disc.computeMatrix()
                u0 = disc.initsolution(b)

                #
                # print("A", A.shape)
                # print("b", b.shape)
                # print("u0", u0.shape)
                # print("fem.nunknowns()", disc.fem.nunknowns())

                m.As.append(A)
                m.B.update(A=A)

                x = m.B.solve(b=b.flatten(), x0=u0.flatten())
                u = b.from_flat_like(x)

                post = disc.postProcess(u)
                scal = post.get("scalar", {})

                row = {
                    "method": name,
                    "level": level,
                    "N": A.shape[0],
                    "niter": getattr(m.B, "niter", None),
                    "res": np.linalg.norm(A @ x - b),
                    **scal,
                }

                if self.callback is not None:
                    self.callback(method=m, disc=disc, level=level, u=u, post=post, row=row)

                rows.append(row)

                if level + 1 < self.nref:
                    # global refinement first
                    marked = np.ones(disc.mesh.ncells, dtype=bool)
                    mesh2, info = m.mesh_hierarchy.refine_nvb(marked)
                    disc2 = m.discretize(mesh2)
                    transfer = disc.build_transfer_to_refined_mesh(info, disc_fine=disc2)
                    m.transfers.append(transfer)
                    u2 = transfer.interpolate(u)
                    disc2.u0 = u2
                    m.discs.append(disc2)

        self.df = pd.DataFrame(rows)
        return self.df

    def print(self):
        if self.df is None:
            raise RuntimeError("run() first")
        print(self.df.to_string())

    def plot_errors(self, rate_ignore=2):
        if self.df is None:
            raise RuntimeError("run() first")

        from Utility.plotting import plot_error_curves
        import matplotlib.pyplot as plt

        df = self.df
        quantities = [
            q for q in ["err_L2c", "err_L2n", "err_H1", "err_Flux"]
            if q in df.columns
        ]

        plot_dicts = {}

        for q in quantities:
            xs = []
            ys = {}

            for method, d in df.groupby("method"):
                d = d.sort_values("N")
                xs.append(d["N"].to_numpy())
                ys[method] = d[q].to_numpy()

            plot_dicts[q] = {
                "x": xs,
                "y": ys,
                "xlabel": "N",
                "ylabel": q,
            }

        plot_error_curves(plot_dicts, rate_ignore=2, separate=True)
        plt.show()

    def plot_iterations(self):
        if self.df is None:
            raise RuntimeError("run() first")

        from Utility.plotting import plot_solutions
        import matplotlib.pyplot as plt

        df = self.df
        if "niter" not in df.columns:
            return

        plot_dicts = {
            "Iterations": {
                "x": [],
                "y": {},
                "xlabel": "N",
                "ylabel": "niter",
            }
        }

        for method, d in df.groupby("method"):
            d = d.sort_values("N")
            plot_dicts["Iterations"]["x"].append(d["N"].to_numpy())
            plot_dicts["Iterations"]["y"][method] = d["niter"].to_numpy()

        plot_solutions(plot_dicts)
        plt.show()