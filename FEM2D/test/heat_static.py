from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(root))

from FEM2D.models import Model, EllipticDiscretization, Application

from Utility import timer


#------------------------------------------------------------------------
class HeatExample(Application):
    def defineProblemData(self, problemdata):
        problemdata.bdrycond.set(type="Dirichlet", colors=[1000, 3000])
        problemdata.bdrycond.set(type="Neumann", colors=[1001, 1002, 1003])

        problemdata.bdrycond.fct[1000] = lambda x, y, z: 200
        problemdata.bdrycond.fct[3000] = lambda x, y, z: 320

        problemdata.postproc.set(name="bdrymean_right", type="bdry_mean", colors=1001)
        problemdata.postproc.set(name="bdrymean_left", type="bdry_mean", colors=1003)
        problemdata.postproc.set(name="bdrymean_up", type="bdry_mean", colors=1002)
        problemdata.postproc.set(name="bdrynflux", type="bdry_nflux", colors=[3000])

        problemdata.params.scal_celllabels["kheat"] = {
            100: 100.01,
            200: 100.0,
        }
        problemdata.params.fct_glob["convection"] = ["0", "0.02"]

    def defineGeometry(self, geom, h, boundary_projectors):
        h=0.05
        h = 0.2
        holes = []

        rectangle = geom.add_rectangle(
            xmin=-1.5, xmax=-0.5,
            ymin=-1.5, ymax=-0.5,
            z=0,
            mesh_size=h,
        )
        geom.add_physical(rectangle.surface, label="200")
        geom.add_physical(rectangle.lines, label="20")
        holes.append(rectangle)

        circle = self.add_circle(
            geom,
            boundary_projectors,
            label=3000,
            center=[0, 0],
            radius=0.5,
            mesh_size=h,
        )

        holes.append(circle)

        p = geom.add_rectangle(
            xmin=-2, xmax=2,
            ymin=-2, ymax=2,
            z=0,
            mesh_size=h,
            holes=holes,
        )
        geom.add_physical(p.surface, label="100")

        for i, line in enumerate(p.lines):
            geom.add_physical(line, label=f"{1000 + i}")


#------------------------------------------------------------------------
class CDR(HeatExample):
    def defineProblemData(self, problemdata):
        super().defineProblemData(problemdata)
        problemdata.params.scal_glob['reaction'] = [1.1]

class NonlinearCDR(HeatExample):
    def defineProblemData(self, problemdata):
        super().defineProblemData(problemdata)

        _coef=0.1

        def reaction(u):
            return 1.1 * u + _coef * u**3

        def reaction_d(u):
            return 1.1 + 3*_coef * u**2

        problemdata.params.fct_glob["reaction"] = reaction
        problemdata.params.fct_glob["reaction_d"] = reaction_d

#------------------------------------------------------------------------
def linear_example():
    linear_solver_params={
        "smoother": "ilu",
        "smoother_kwargs": {
            "drop_tol": 0.01,
            "fill_factor": 1.1,
            "permc_spec": "NATURAL",
            "diag_pivot_thresh": 0.0,
        },
        "nu_pre": 2,
        "nu_post": 2,
        "gamma": 1.0,
        "acceleration": "arnoldi",
        "arnoldi_dim": 3,
    }
    heat = Model(
        application=HeatExample(),
        discretization=EllipticDiscretization,
        fem="p1",
        linear_solver="geommg",
        linear_solver_params=linear_solver_params,
        disc_params={"convmethod": "centered"},
    )

    mesh_timer = timer.Timer()

    heat.afem_loop(
        niter=7,
        theta = 0.9,
        plot_solution=True,
        mesh_timer=mesh_timer,
    )

    print(heat.timer.summary_by_leaf() + "\n")
    print(mesh_timer.summary_by_leaf())

    print([level.mesh.nnodes for level in heat.mesh_hierarchy.levels])
    print([level.mesh.ncells for level in heat.mesh_hierarchy.levels])


#------------------------------------------------------------------
def newton(NewtonDriver):
    from Newton import newton, armijo, newtondata
    heat = Model(
        application=NonlinearCDR(),
        discretization=EllipticDiscretization,
        fem="p1",
        disc_params={"dirichletmethod":"nitsche",
                     "reaction_lumped": False,
                     "convmethod": "lps",
                     "lpsparam": 1.0},
        linear_solver="geommg",
    )


    x0 = heat.initial_guess()
    newton = newton.Newton(
        nd = NewtonDriver(heat, debug=False, theta=0.8, max_meshiter=100),
        verbose=2,
        globalization=armijo.ArmijoGlobalization(
            omega=0.5,
            maxiter=10,
            c=1e-4,
            debug=False,
        ),
        sdata=newtondata.StoppingParamaters(
            forcing_lambda=0.5,
            forcing_kappa=0.75,
            rtol=0.001,
        )
    )
    xs, info, logger = newton.solve(x0)

    if not info.success:
        print(info.success,info.failure)
    else:
        print('---- time ---')
        print(heat.timer.summary(),'\n')
        logger.print_history()

    # print(f"{heat.discs[-1]=}")
    if info.success:
        heat.plot_solution(xs)
    else:
        disc = heat.discs[-1]
        heat.plot_solution(u=getattr(disc, "u0", xs), disc=disc)

#------------------------------------------------------------------
# linear_example()

one_level=False
if one_level:
    from FEM2D.models.newton_driver_one_level import NewtonDriverOneLevel as NewtonDriver
else:
    from FEM2D.models.newton_driver_afem import NewtonDriverAFEM as NewtonDriver
newton(NewtonDriver)
