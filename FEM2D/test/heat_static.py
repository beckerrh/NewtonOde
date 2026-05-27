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
from Utility import timer

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
linearsolver = {'method':'pyamg', 'disp':1}
# linearsolver = "spsolve"
heat = Elliptic(application=HeatExample(), fem='p1', disc_params=disc_params, linearsolver=linearsolver)
plotting = True
mesh_timer = timer.Timer()
for ell in range(13):
    with heat.timer.scope(f"AFEM{ell:02d}"):
        with heat.timer("solve"):
            result, u = heat.static(method="linear")
        print(f"{result.info=}")
        for k, v in result.data['scalar'].items():
            print(f"{k:20s} : {v}")
        eta = result.data['cell']['eta']
        if plotting:
            with heat.timer("plot"):
                fig = plt.figure(figsize=(10, 8))
                fig.suptitle(f"{heat.application.__class__.__name__} nn={heat.mesh.nnodes:7d} ({ell=} )", fontsize=16)
                outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
                heat.mesh.plot(fig=fig, outer=outer[0], bdry=True)
                data = u.tovisudata()
                data.setdefault("cell", {})
                data["cell"]["k"] = heat.kheatcell
                data["cell"]["eta"] = eta
                heat.mesh.plot(data=data, alpha=0.5, fig=fig, outer=outer[1])
                plt.show()
        with heat.timer("marking"):
            marked = marking.dorfler_marking(eta, theta=0.75)
        with heat.timer("refine"):
            mesh2, info = heat.mesh.refine_nvb(marked, timer=mesh_timer, debug=False)
        with heat.timer("setMesh"):
            heat.setMesh(mesh2)  # mandatory: rebuild fems, bdry data, coefficients, matrix cache
# print(result.info['timer'].summary()+'\n')
print(heat.timer.summary_by_leaf()+'\n')
print(mesh_timer.summary_by_leaf())
