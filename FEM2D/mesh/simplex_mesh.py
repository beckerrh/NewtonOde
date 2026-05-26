# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse

from Utility import timer
from . import topology, geometry


class SimplexMesh:
    """
    Simplicial mesh container.

    Public compatibility attributes kept:
        cells, facesOfCells, cellsOfFaces, pointsc, pointsf, dV
        bdrylabels, cellsoflabel, linesoflabel, verticesoflabel
    """

    def __init__(self, points, cells=None, *, labels=None):
        self.timer = timer.Timer()

        # Backward compatibility:
        # old code used SimplexMesh(meshio_mesh)
        if cells is None:
            from .mesh_io import from_meshio
            other = from_meshio(points)
            self.__dict__.update(other.__dict__)
            return

        self.points = np.asarray(points, dtype=float)
        self.cells = np.asarray(cells, dtype=np.int64)

        if self.points.ndim != 2:
            raise ValueError(f"points must be 2D, got {self.points.shape=}")

        if self.points.shape[1] == 2:
            self.points = np.column_stack(
                [self.points, np.zeros(self.points.shape[0])]
            )

        if self.points.shape[1] != 3:
            raise ValueError(f"points must have 2 or 3 columns, got {self.points.shape=}")

        self.dimension = self.cells.shape[1] - 1
        self.nnodes = self.points.shape[0]

        self.bdrylabels = {}
        self.cellsoflabel = {}
        self.linesoflabel = {}
        self.verticesoflabel = {}

        if labels is not None:
            self.__dict__.update(labels)

        self._rebuild()
        self.check()

    @classmethod
    def from_meshio(cls, mesh):
        from .meshio_io import from_meshio
        return from_meshio(mesh)

    def _rebuild(self):
        topology.construct_faces_from_cells(self)
        with self.timer("construct_faces_from_cells"):
            self.ncells = self.cells.shape[0]
            self.nfaces = self.faces.shape[0]
            geometry.construct_centers(self)
            geometry.construct_normals_and_volumes(self)
        with self.timer("construct_geometry"):
            topology.construct_inner_faces(self)
            self._set_compatibility_aliases()

    def _set_compatibility_aliases(self):
        self.facesOfCells = self.faces_of_cells
        self.cellsOfFaces = self.cells_of_faces
        self.pointsc = self.cell_centers
        self.pointsf = self.face_centers
        self.dV = self.cell_volumes

    def finalize_after_topology_change(self):
        self.points = np.asarray(self.points)
        self.cells = np.asarray(self.cells, dtype=int)
        self.nnodes = self.points.shape[0]
        self.ncells = self.cells.shape[0]

        self._rebuild()

        if hasattr(self, "celllabels"):
            self.cellsoflabel = {}
            for icell, label in enumerate(self.celllabels):
                self.cellsoflabel.setdefault(label, []).append(icell)

            self.cellsoflabel = {
                label: np.asarray(ids, dtype=int)
                for label, ids in self.cellsoflabel.items()
            }
    def check(self):
        used = np.unique(self.cells)
        if len(used) != self.nnodes:
            raise ValueError(f"{len(used)=} BUT {self.nnodes=}")
        if not np.all(used == np.arange(self.nnodes)):
            raise ValueError("Cell vertex numbering must be contiguous from 0 to nnodes-1.")

    def getBdryPoints(self, colors):
        if not isinstance(colors, (list, tuple)):
            colors = [colors]
        bdrypoints = []
        for color in colors:
            if not isinstance(color, int):
                color = self.labeldict_s2i[color]
            facesdir = self.bdrylabels[color]
            bdrypoints.append(np.unique(self.faces[facesdir].ravel()))
        return np.array(bdrypoints).reshape(-1)

    def bdryFaces(self, colors=None):
        if colors is None:
            colors = self.bdrylabels.keys()
        pos = [0]
        for color in colors:
            pos.append(pos[-1] + len(self.bdrylabels[color]))

        faces = np.empty(pos[-1], dtype=np.uint32)
        for i, color in enumerate(colors):
            faces[pos[i]:pos[i + 1]] = self.bdrylabels[color]
        return faces

    def facesOfCellsNotOnInnerFaces(self, ci0, ci1):
        faces = self.faces[self.innerfaces]
        fi0_bis = np.empty_like(faces)
        fi1_bis = np.empty_like(faces)
        for i in range(faces.shape[1]):
            fi0_bis[:, i] = self.facesOfCells[ci0][
                self.cells[ci0] == faces[:, i][:, None]
            ]
            fi1_bis[:, i] = self.facesOfCells[ci1][
                self.cells[ci1] == faces[:, i][:, None]
            ]
        return fi0_bis, fi1_bis

    def computeSimpOfVert(self, test=False):
        S = sparse.dok_matrix((self.nnodes, self.ncells), dtype=int)
        for ic in range(self.ncells):
            S[self.cells[ic, :], ic] = ic + 1
        S = S.tocsr()
        S.data -= 1
        self.simpOfVert = S

    def write(self, filename, dirname=None, data=None):
        from .meshio_io import write
        return write(self, filename, dirname=dirname, data=data)

    def writemeshio(self, filename, dirname=None, data=None):
        from .meshio_io import writemeshio
        return writemeshio(self, filename, dirname=dirname, data=data)

    def plot(self, **kwargs):
        from . import plotmesh
        if kwargs.pop("bdry", False):
            return plotmesh.meshWithBoundaries(self, **kwargs)
        return plotmesh.meshWithData(self, **kwargs)

    def __repr__(self):
        s = f"dim/nnodes/nfaces/ncells: {self.dimension}/{self.nnodes}/{self.nfaces}/{self.ncells}"
        if hasattr(self, "labeldict_i2s"):
            s += f"\nbdrylabels={[self.labeldict_i2s.get(k, k) for k in self.bdrylabels.keys()]}"
            s += f"\ncellsoflabel={[self.labeldict_i2s.get(k, k) for k in self.cellsoflabel.keys()]}"
        else:
            s += f"\nbdrylabels={list(self.bdrylabels.keys())}"
            s += f"\ncellsoflabel={list(self.cellsoflabel.keys())}"
        return s

    def __str__(self):
        return f"dim/nnodes/nfaces/ncells: {self.dimension}/{self.nnodes}/{self.nfaces}/{self.ncells}"