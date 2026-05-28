import os
import numpy as np
import meshio

from .simplex_mesh import SimplexMesh
from . import gmsh_labels


def simplex_names_from_meshio(mesh):
    celltypes = [c.type for c in mesh.cells]

    if "tetra" in celltypes:
        return 3, "tetra", "triangle"
    if "triangle" in celltypes:
        return 2, "triangle", "line"
    if "line" in celltypes:
        return 1, "line", "vertex"

    raise ValueError(f"No simplex cells found in {celltypes=}")


def from_meshio(mesh):
    dim, simplex_name, face_name = simplex_names_from_meshio(mesh)

    cells = np.asarray(mesh.cells_dict[simplex_name], dtype=np.int64)

    used = np.unique(cells)
    if not np.all(used == np.arange(len(used))):
        msg = (
            f"Dangling or non-contiguous points are not supported yet.\n"
            f"{len(used)=}, {mesh.points.shape=}, {used=}"
        )
        raise ValueError(msg)

    points = mesh.points[: len(used)]

    smesh = SimplexMesh(points, cells)

    smesh.pygmsh = mesh
    smesh.cellsname = simplex_name
    smesh.topology.facesname = face_name
    smesh.topology.facesdata = np.asarray(mesh.cells_dict[face_name], dtype=np.int64)

    gmsh_labels.attach_gmsh_labels(
        smesh,
        meshio_mesh=mesh,
        simplex_name=simplex_name,
        face_name=face_name,
    )

    return smesh


def _getfilename(filename, dirname=None):
    if dirname is not None:
        dirname = dirname + os.sep + "mesh"
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        filename = os.path.join(dirname, filename)
    return filename


def writemeshio(mesh, filename, dirname=None, data=None):
    filename = _getfilename(filename, dirname)
    mesh.pygmsh.write(filename, file_format="gmsh22")


def write(mesh, filename, dirname=None, data=None):
    filename = _getfilename(filename, dirname)

    if mesh.dimension == 1:
        cell_name = "line"
    elif mesh.dimension == 2:
        cell_name = "triangle"
    elif mesh.dimension == 3:
        cell_name = "tetra"
    else:
        raise ValueError(f"Unsupported dimension {mesh.dimension}")

    args = {
        "points": mesh.points,
        "cells": {cell_name: mesh.cells},
    }

    if data is not None:
        if "point" in data:
            args["point_data"] = data["point"]
        if "cell" in data:
            args["cell_data"] = {k: [data["cell"][k]] for k in data["cell"].keys()}

    meshio.write(filename, meshio.Mesh(**args))