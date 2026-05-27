import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from FEM2D.models.elliptic import Elliptic
from FEM2D.models.application import Application
from FEM2D.mesh import marking
from FEM2D.mesh.mesh_hierarchy import MeshHierarchy
from FEM2D.linalg.solver_factory import getLinearSolver
from FEM2D.fems import mesh_transfer, transfer_p1

from Utility import timer
from types import SimpleNamespace


# define Application class
#--------------------------------------------------------------
class HeatExample(Application):
    def defineProblemData(self, problemdata):
        # fill problem data
        # boundary conditions
        problemdata.bdrycond.set(type="Dirichlet", colors=[1000, 3000])
        problemdata.bdrycond.set(type="Neumann", colors=[1001, 1002, 1003])
        problemdata.bdrycond.fct[1000] = lambda x, y, z: 200
        problemdata.bdrycond.fct[3000] = lambda x, y, z: 320
        # postprocess
        problemdata.postproc.set(name='bdrymean_right', type='bdry_mean', colors=1001)
        problemdata.postproc.set(name='bdrymean_left', type='bdry_mean', colors=1003)
        problemdata.postproc.set(name='bdrymean_up', type='bdry_mean', colors=1002)
        problemdata.postproc.set(name='bdrynflux', type='bdry_nflux', colors=[3000])
        # paramaters in equation
        problemdata.params.set_scal_cells("kheat", [100], 0.001)
        problemdata.params.set_scal_cells("kheat", [200], 10.0)
        problemdata.params.fct_glob["convection"] = ["0", "0.002"]
    def defineGeometry(self, geom, h):
        holes = []
        rectangle = geom.add_rectangle(xmin=-1.5, xmax=-0.5, ymin=-1.5, ymax=-0.5, z=0, mesh_size=h)
        geom.add_physical(rectangle.surface, label="200")
        geom.add_physical(rectangle.lines, label="20")  # required for correct boundary labels (!?)
        holes.append(rectangle)
        circle = geom.add_circle(x0=[0, 0], radius=0.5, mesh_size=h, num_sections=6, make_surface=False)
        geom.add_physical(circle.curve_loop.curves, label="3000")
        holes.append(circle)
        p = geom.add_rectangle(xmin=-2, xmax=2, ymin=-2, ymax=2, z=0, mesh_size=h, holes=holes)
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")

#--------------------------------------------------------------
disc_params={'dirichletmethod':'nitsche'}
heat = Elliptic(application=HeatExample(), fem='p1', disc_params=disc_params)
mesh0 = heat.application.createMesh()
hierarchy = MeshHierarchy(mesh0)
transfers = []
As = []
with heat.timer("create_discretization"):
    discs = [heat.discretize(mesh0)]
plotting = False
mesh_timer = timer.Timer()
linear_solver = 'pyamg'
linear_solver = 'geommg'
B = getLinearSolver(
    method=linear_solver,
    As=As,
    transfers=transfers,
)
for ell in range(13):
    disc = discs[-1]
    with heat.timer.scope(f"AFEM{ell:02d}"):
        with heat.timer("rhs"):
            b = disc.computeRhs()
            u0 = disc.initsolution(b)
        with heat.timer("matrix"):
            A = disc.computeMatrix()
            As.append(A)
        with heat.timer("linear_solver"):
            B.update(A=A)
            u = B.solve(b=b, x0=u0)
        res = np.linalg.norm(A @ u - b)
        print(
            f"{ell:2d} "
            f"N={A.shape[0]:7d} "
            f"niter={B.niter:2d} "
            f"res={res:.3e}"
        )
        with heat.timer("postproc"):
            postproc = disc.postProcess(u)

            # self.save(u=u)
        result = SimpleNamespace(u=u, postproc=postproc)
        for k, v in result.postproc['scalar'].items():
            print(f"{k:20s} : {v}")
        eta = result.postproc['cell']['eta']
        if plotting:
            with heat.timer("plot"):
                fig = plt.figure(figsize=(10, 8))
                fig.suptitle(f"{heat.application.__class__.__name__} nn={disc.mesh.nnodes:7d} ({ell=} )", fontsize=16)
                outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
                disc.mesh.plot_boundary(fig=fig, outer=outer[0])
                data = {"point": {"u": result.u}, "cell": {'k': disc.kheatcell, 'eta':eta}, "global": {}}
                disc.mesh.plot(data=data, alpha=0.5, fig=fig, outer=outer[1])
                plt.show()
        with heat.timer("marking"):
            marked = marking.dorfler_marking(eta, theta=0.9)
        with heat.timer("refine"):
            mesh2, info = hierarchy.refine_nvb(marked, timer=mesh_timer, debug=False)
            # mesh2, info = heat.mesh.refine_nvb(marked, timer=mesh_timer, debug=False)
        with heat.timer("interpolate"):
            u2 = mesh_transfer.interpolate_p1_to_refined_mesh(info, result.u)

            P = mesh_transfer.p1_prolongation(info)
            transfers.append(transfer_p1.P1Transfer(P))
        if plotting:
            with heat.timer("plot_interpolation"):
                fig = plt.figure(figsize=(10, 8))
                fig.suptitle("P1 interpolation after NVB refinement", fontsize=16)
                outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
                data = {"point": {"u": result.u}, "cell": {}, "global": {}}
                disc.mesh.plot(data=data, fig=fig, outer=outer[0], alpha=0.1)
                data2 = {"point": {"u": u2}, "cell": {}, "global": {}}
                mesh2.plot(data=data2, fig=fig, outer=outer[1])
                plt.show()
        with heat.timer("create_discretization"):
            discs.append(heat.discretize(mesh2))
# print(result.info['timer'].summary()+'\n')
print(heat.timer.summary_by_leaf()+'\n')
print(mesh_timer.summary_by_leaf())
print([level.mesh.nnodes for level in hierarchy.levels])
print([level.mesh.ncells for level in hierarchy.levels])