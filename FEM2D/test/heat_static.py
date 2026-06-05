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

        problemdata.params.set_scal_cells("kheat", [100], 0.001)
        problemdata.params.set_scal_cells("kheat", [200], 10.0)
        problemdata.params.fct_glob["convection"] = ["0", "0.02"]

    def defineGeometry(self, geom, h, boundary_projectors):
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
def linear_example():
    linear_solvers=["pyamg", 'spsolve', "geommg"]
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
def linear_example_by_newton():
    from FEM2D.models.newton_driver import NewtonDriver
    from Newton import newton, newtondata
    heat = Model(
        application=HeatExample(),
        discretization=EllipticDiscretization,
        fem="p1",
        disc_params={"dirichletmethod":"nitsche"},
    )


    sdata = newtondata.StoppingParamaters(maxiter=50, rtol=1e-3, forcing_kappa=0.5)
    x0 = heat.initial_guess()
    newton = newton.Newton(
        nd = NewtonDriver(heat),
        verbose=2,
        sdata=sdata,
    )
    xs, info, logger = newton.solve(x0)

    if not info.success:
        print(info.success,info.failure)
    else:
        print('---- time ---')
        print(heat.timer.summary(),'\n')
        # print(heat.B.timer.summary(),'\n')
        # heat.plot_eta()
        logger.print_history()



#------------------------------------------------------------------
linear_example_by_newton()
