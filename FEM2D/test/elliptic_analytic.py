import numpy as np
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(root))


from FEM2D.models import Model, EllipticDiscretization
from FEM2D.mesh import testmeshes
import FEM2D.models.application
from Utility.compare_methods import CompareMethods

#----------------------------------------------------------------#
class EllipticApplicationWithExactSolution(FEM2D.models.application.Application):
    def __init__(self, dim, exactsolution, **kwargs):
        if dim == 1:
            self.defineGeometry = testmeshes.unitline
            colors = [10000,10001]
        elif dim == 2:
            self.defineGeometry = testmeshes.add_unitsquare
            colors = [1000, 1001, 1002, 1003]
            # colorsrob = [1002]
            # colorsneu = [1001]
        else:
            self.defineGeometry = testmeshes.unitcube
            colors = [100, 101, 102, 103, 104, 105]
            # colorsrob = [101]
            # colorsneu = [103]
        self.colors = colors
        super().__init__(exactsolution=exactsolution, dimension=dim)

    def defineProblemData(self, problemdata):
        data = problemdata
        colors = self.colors
        colorsneu, colorsrob = [], []
        colorsdir = [col for col in colors if col not in colorsrob and col not in colorsneu]
        data.bdrycond.set("Dirichlet", colorsdir)
        data.bdrycond.set("Neumann", colorsneu)
        data.bdrycond.set("Robin", colorsrob)
        for col in colorsrob: data.bdrycond.param[col] = 100.
        data.params.scal_glob['kheat'] = 0.1
        data.params.fct_glob['convection'] = ['0.8', '1.1']
        data.params.scal_glob['reaction'] = [1.1]
        # data.params.scal_glob['reaction'] = [
        #     [0.8, -0.2],
        #     [0.5, 1.1],
        # ]


#================================================================#
if __name__ == '__main__':

    exactsolution = ["Quadratic", "Linear"]
    exactsolution = ["Quadratic"]
    app = EllipticApplicationWithExactSolution(dim=2, exactsolution=exactsolution)

    print(f"{app.ncomps=}")


    def method(fem, **disc_params):
        return {
            "discretization": EllipticDiscretization,
            "fem": fem,
            "linear_solver": "geommg",
            "disc_params": disc_params,
        }


    methods = {
        "P1 strong": method("p1", dirichletmethod="strong"),

        "P1 nitsche": method(
            "p1",
            dirichletmethod="nitsche",
            nitscheparam=10,
        ),

        "CR1 strong": method(
            "cr1",
            dirichletmethod="strong",
        ),

        "CR1 nitsche": method(
            "cr1",
            dirichletmethod="nitsche",
            nitscheparam=10,
        ),

        "CR1 nitsche lumped": method(
            "cr1",
            dirichletmethod="nitsche",
            nitscheparam=10,
            nitsche_lumped=True,
        ),
    }

    def callback(method, disc, level, u, post, row):
        A = method.As[-1]
        ncomp = disc.application.ncomps[0]
        print("ncomps", disc.application.ncomps)
        print("A", A.shape)
        print("u", u)
        print("ndof/ncomp", A.shape[0] // ncomp)
        print("remainder", A.shape[0] % ncomp)
        assert A.shape[0] % ncomp == 0
        assert u.size == A.shape[0]
        assert u.nparts == 1

    cmp = CompareMethods(
        application=app,
        model=Model,
        methods=methods,
        nref=8,
        # callback=callback,
    )

    df = cmp.run()
    cmp.print()

    plotting = True

    if plotting:
        cmp.plot_errors(rate_ignore=2)
        cmp.plot_iterations()
