import pygmsh
from FEM2D.models.problemdata import ProblemData
from FEM2D.mesh.simplex_mesh import SimplexMesh

# ================================================================ #
class Application:
    def __init__(self, **kwargs):
        self.h = kwargs.pop('h', 0.5)
        self.exactsolution = kwargs.pop('exactsolution', None)
        self.random_exactsolution = kwargs.pop('random_exactsolution', None)
        self.generatePDforES = kwargs.pop('generatePDforES', None)
        if self.generatePDforES is None and self.exactsolution:
            self.generatePDforES = True
        self.problemdata = ProblemData()
        self.defineProblemData(self.problemdata)
        # print(f"{self.problemdata=}")
        scal_glob = kwargs.pop('scal_glob', {})
        for k,v in scal_glob.items():
            self.problemdata.params.scal_glob[k] = v
        if len(kwargs.keys()):
            raise ValueError(f"*** unused arguments {kwargs=}")

    def defineGeometry(self, geom, h): raise ValueError(f"not written")
    def createExactSolution(self, mesh, ncomps):
        dim, ran = mesh.dimension, self.random_exactsolution
        assert isinstance(ncomps, (list,tuple))
        if isinstance(self.exactsolution, str): names=[self.exactsolution]
        else: names = self.exactsolution
        # print(f"***{ncomps=} {names=} {self.exactsolution=}")
        assert len(ncomps) == len(names)
        self.exactsolution = []
        for i in range(len(ncomps)):
            print(f"{i=} {names[i]=} {ncomps[i]=}")
            self.exactsolution.append(analyticalSolution(names[i], dim, ncomps[i], ran))
        return
        # print(f"****** createExactSolution: {dim=} {ncomp=} {self.exactsolution=}")
        # if isinstance(ncomp, (list,tuple)) or isinstance(self.exactsolution,(list,tuple)):
        #     assert len(ncomp)==len(self.exactsolution)
        #     es= []
        #     for i in range(len(ncomp)):
        #         es.append(analyticalSolution(self.exactsolution[i], dim, ncomp[i], ran))
        #     self.exactsolution = es
        # else:
        #     if isinstance(self.exactsolution, str):
        #         self.exactsolution = analyticalSolution(self.exactsolution, dim, ncomp, ran)
    def createMesh(self, h=0.5):
        if h is None: h = self.h
        with pygmsh.geo.Geometry() as geom:
            self.defineGeometry(geom, h)
            mesh = geom.generate_mesh()
        return SimplexMesh.from_meshio(mesh)
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
