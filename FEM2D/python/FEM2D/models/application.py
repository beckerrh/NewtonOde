import pygmsh
from .problemdata import ProblemData
from ..mesh import SimplexMesh
from ..mesh.boundary_geometry import CircleProjector
from Utility.analyticalfunction import  AnalyticalFunction, analytical_solution

# ================================================================ #
class Application:
    def __init__(self, **kwargs):
        self.h = kwargs.pop('h', 0.5)
        self.ncomps = kwargs.pop("ncomps", None)
        self.exactsolution_spec = kwargs.pop('exactsolution', None)
        self.exactsolution = None
        if self.ncomps is None:
            if isinstance(self.exactsolution_spec, (list, tuple)):
                self.ncomps = [len(self.exactsolution_spec)]
            else:
                self.ncomps = [1]
        self.random_exactsolution = kwargs.pop('random_exactsolution', None)

        if self.exactsolution_spec is not None:
            if not "dimension" in kwargs:
                raise KeyError(f"Application: for exact_solution needs 'dimension'")
            self.makeExactSolution(kwargs.pop("dimension", None))

        self.problemdata = ProblemData()
        self.defineProblemData(self.problemdata)
        # print(f"{self.problemdata=}")
        scal_glob = kwargs.pop('scal_glob', {})
        for k,v in scal_glob.items():
            self.problemdata.params.scal_glob[k] = v
        if len(kwargs.keys()):
            raise ValueError(f"*** unused arguments {kwargs=}")

    #----------------------------------------------------------------
    def _normalize_exactsolution_blocks(self):
        ex = self.exactsolution_spec
        ncomps = self.ncomps

        if ex is None:
            return None

        # ncomps=[2], exactsolution=["Quadratic", "Linear"]
        # means one vector unknown with two scalar components.
        if len(ncomps) == 1:
            return [ex]

        # ncomps=[2,1], exactsolution=[["u1", "u2"], "p"]
        # means several unknown blocks.
        if not isinstance(ex, (list, tuple)):
            raise ValueError("For several unknown blocks, exactsolution must be a list/tuple")

        if len(ex) != len(ncomps):
            raise ValueError(f"{len(ex)=} != {len(ncomps)=}")

        return list(ex)

    #----------------------------------------------------------------
    def makeExactSolution(self, dim):
        if self.exactsolution is not None:
            return
        print(f"{dim=} {self.exactsolution_spec=}")
        ran = self.random_exactsolution
        ncomps = self.ncomps

        blocks = self._normalize_exactsolution_blocks()
        if blocks is None:
            return

        exact = []
        for block, nc in zip(blocks, ncomps):
            exact.append(analytical_solution(block, dim, nc, ran))

        self.exactsolution = exact

    #----------------------------------------------------------------
    def add_circle(
            self,
            geom,
            boundary_projectors,
            *,
            label,
            center,
            radius,
            mesh_size,
            num_sections=6,
            make_surface=False,
    ):
        circle = geom.add_circle(
            x0=center,
            radius=radius,
            mesh_size=mesh_size,
            num_sections=num_sections,
            make_surface=make_surface,
        )

        geom.add_physical(circle.curve_loop.curves, label=str(label))

        proj = CircleProjector(center=center, radius=radius)

        boundary_projectors[label] = proj
        boundary_projectors[str(label)] = proj

        return circle

    def createMesh(self, h=0.5):
        if h is None:
            h = self.h
        boundary_projectors = {}
        with pygmsh.geo.Geometry() as geom:
            self.defineGeometry(geom, h, boundary_projectors=boundary_projectors)
            meshio_mesh = geom.generate_mesh()
        mesh = SimplexMesh.from_meshio(meshio_mesh)
        if boundary_projectors:
            from FEM2D.mesh.boundary_geometry import LabelBoundaryProjector
            mesh.geometry.boundary_projector = LabelBoundaryProjector(boundary_projectors)
        return mesh
    def defineProblemData(self, problemdata):
        pass
    def plot(self, mesh, data, **kwargs):
        if mesh.dimension != 2:
            raise ValueError("not written")
        import matplotlib.pyplot as plt
        from matplotlib.figure import figaspect
        import matplotlib.gridspec as gridspec
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig = kwargs.pop('fig', None)
        gs = kwargs.pop('gs', None)
        # print(f"{data=}")
        nplots = len(data['cell'].keys()) + len(data['point'].keys())
        if fig is None:
            if gs is not None:
                raise ValueError(f"got gs but no fig")
            fig = plt.figure(constrained_layout=True, figsize=figaspect(nplots))
            # appname = kwargs.pop('title', self.__class__.__name__)
            # fig.set_title(f"{appname}")
        if gs is None:
            gs = fig.add_gridspec(1, 1)[0,0]
        inner = gridspec.GridSpecFromSubplotSpec(nrows=nplots, ncols=1, subplot_spec=gs, wspace=0.3, hspace=0.3)
        x, y, tris = mesh.geometry.points[:,0], mesh.geometry.points[:,1], mesh.cells
        iplot = 0
        for name,values in data['cell'].items():
            ax = fig.add_subplot(inner[iplot])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.triplot(x, y, tris, color='gray', lw=1, alpha=0.1)
            cnt = ax.tripcolor(x, y, tris, facecolors=values, edgecolors='k', cmap='jet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.4)
            clb = plt.colorbar(cnt, cax=cax, orientation='vertical')
            clb.ax.set_title(name)
            iplot += 1
        for name,values in data['point'].items():
            # print(f"{name=} {values.min()=}  {values.max()=}")
            ax = fig.add_subplot(inner[iplot])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.triplot(x, y, tris, color='gray', lw=1, alpha=0.1)
            cnt = ax.tricontourf(x, y, tris, values, levels=16, cmap='jet', alpha=1.)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.4)
            clb = plt.colorbar(cnt, cax=cax, orientation='vertical')
            clb.ax.set_title(name)
            iplot += 1
