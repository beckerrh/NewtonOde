# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse

from contextlib import nullcontext
from dataclasses import dataclass, field

from . import backend_dispatch

@dataclass
class MeshLabels:
    boundary: dict = field(default_factory=dict)
    cell: dict = field(default_factory=dict)
    line: dict = field(default_factory=dict)
    vertex: dict = field(default_factory=dict)
    names: dict = field(default_factory=dict)

@dataclass
class MeshTopology:
    cells: np.ndarray | None = None
    faces: np.ndarray | None = None
    faces_of_cells: np.ndarray | None = None
    cells_of_faces: np.ndarray | None = None
    inner_faces: np.ndarray | None = None

@dataclass
class MeshGeometry:
    points: np.ndarray | None = None
    cell_centers: np.ndarray | None = None
    face_centers: np.ndarray | None = None
    cell_volumes: np.ndarray | None = None
    normals: np.ndarray | None = None
    boundary_projector = None

class SimplexMesh:
    """
    Simplicial mesh container.
    """

    def __init__(self, points, cells, *, labels=None, rebuild=True, check=True):
        self.labels = MeshLabels()
        self.topology = MeshTopology()
        self.topology.cells = np.asarray(cells, dtype=np.int64)
        self.geometry = MeshGeometry()

        self.geometry.points = np.asarray(points, dtype=float)

        if self.geometry.points.ndim != 2:
            raise ValueError(f"points must be 2D, got {self.geometry.points.shape=}")

        if self.geometry.points.shape[1] == 2:
            self.geometry.points = np.column_stack(
                [self.geometry.points, np.zeros(self.geometry.points.shape[0])]
            )

        if self.geometry.points.shape[1] != 3:
            raise ValueError(
                f"points must have 2 or 3 columns, got {self.geometry.points.shape=}"
            )

        self.nnodes = self.geometry.points.shape[0]
        self.ncells = self.topology.cells.shape[0]
        self.dimension = self.topology.cells.shape[1] - 1

        if labels is not None:
            self.labels.boundary = labels.get("bdrylabels", {})
            self.labels.cell = labels.get("cellsoflabel", {})
            self.labels.line = labels.get("linesoflabel", {})
            self.labels.vertex = labels.get("verticesoflabel", {})
            self.labels.names = labels.get("names", {})

        if rebuild:
            self.rebuild_mesh()

        if check:
            self.check()


    @classmethod
    def from_meshio(cls, mesh):
        from .mesh_io import from_meshio
        return from_meshio(mesh)

    def refine_nvb(self, marked, debug=False, timer=None):
        return backend_dispatch.refine_nvb(
            self,
            marked,
            debug=debug,
            timer=timer,
        )

    def construct_inner_faces(self):
        from .topology import construct_inner_faces
        construct_inner_faces(self)

    def finalize_after_topology_change(
            self,
            timer=None,
    ):
        self.geometry.points = np.asarray(self.geometry.points)
        self.topology.cells = np.asarray(self.topology.cells, dtype=int)
        self.nnodes = self.geometry.points.shape[0]
        self.ncells = self.topology.cells.shape[0]
        with timer("rebuild") if timer else nullcontext():
            self.rebuild_mesh(
                timer=timer,
            )
        if hasattr(self, "cell_markers"):
            with timer("celllabels") if timer else nullcontext():
                self.labels.cell = self._cell_labels_from_markers(self.cell_markers)

    def _cell_labels_from_markers(self, cell_markers):
        markers = np.asarray(cell_markers, dtype=np.int64)
        order = np.argsort(markers)
        markers_s = markers[order]

        cuts = np.flatnonzero(np.r_[True, markers_s[1:] != markers_s[:-1]])

        labels_cell = {}
        for k, start in enumerate(cuts):
            stop = cuts[k + 1] if k + 1 < len(cuts) else markers_s.size
            label = int(markers_s[start])
            labels_cell[label] = order[start:stop].astype(int, copy=False)

        return labels_cell

    def rebuild_mesh(self, timer=None):
        backend_dispatch.rebuild_mesh(self, timer=timer)


    def check(self):
        used = np.unique(self.topology.cells)
        if len(used) != self.nnodes:
            raise ValueError(f"{len(used)=} BUT {self.nnodes=}")
        if not np.all(used == np.arange(self.nnodes)):
            raise ValueError("Cell vertex numbering must be contiguous from 0 to nnodes-1.")

    def getBdryPoints(self, colors):
        if not isinstance(colors, (list, tuple)):
            colors = [colors]
        bdrypoints = []
        for color in colors:
            # if not isinstance(color, int):
            #     color = self.labeldict_s2i[color]
            facesdir = self.labels.boundary[color]
            bdrypoints.append(np.unique(self.topology.faces[facesdir].ravel()))
        return np.array(bdrypoints).reshape(-1)

    def bdryFaces(self, colors=None):
        if colors is None:
            colors = self.labels.boundary.keys()
        pos = [0]
        for color in colors:
            pos.append(pos[-1] + len(self.labels.boundary[color]))

        faces = np.empty(pos[-1], dtype=np.uint32)
        for i, color in enumerate(colors):
            faces[pos[i]:pos[i + 1]] = self.labels.boundary[color]
        return faces

    def faces_of_cellsNotOnInnerFaces(self, ci0, ci1):
        faces = self.topology.faces[self.topology.inner_faces]
        fi0_bis = np.empty_like(faces)
        fi1_bis = np.empty_like(faces)
        for i in range(faces.shape[1]):
            fi0_bis[:, i] = self.topology.faces_of_cells[ci0][
                self.topology.cells[ci0] == faces[:, i][:, None]
            ]
            fi1_bis[:, i] = self.topology.faces_of_cells[ci1][
                self.topology.cells[ci1] == faces[:, i][:, None]
            ]
        return fi0_bis, fi1_bis

    def computeSimpOfVert(self, test=False):
        S = sparse.dok_matrix((self.nnodes, self.ncells), dtype=int)
        for ic in range(self.ncells):
            S[self.topology.cells[ic, :], ic] = ic + 1
        S = S.tocsr()
        S.data -= 1
        self.simpOfVert = S

    def write(self, filename, dirname=None, data=None):
        from FEM2D.mesh.mesh_io import write
        return write(self, filename, dirname=dirname, data=data)

    def writemeshio(self, filename, dirname=None, data=None):
        from .mesh_io import writemeshio
        return writemeshio(self, filename, dirname=dirname, data=data)

    def plot_boundary(self, **kwargs):
        from . import plotmesh
        return plotmesh.meshWithBoundaries(self, **kwargs)
    def plot(self, **kwargs):
        from . import plotmesh
        return plotmesh.meshWithData(self, **kwargs)

    def __repr__(self):
        s = f"dim/nnodes/nfaces/ncells: {self.dimension}/{self.nnodes}/{self.nfaces}/{self.ncells}"
        s += f"\nbdrylabels={list(self.labels.boundary.keys())}"
        s += f"\ncellsoflabel={list(self.labels.cell.keys())}"
        return s

    def __str__(self):
        return f"dim/nnodes/nfaces/ncells: {self.dimension}/{self.nnodes}/{self.nfaces}/{self.ncells}"