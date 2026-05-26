# -*- coding: utf-8 -*-
import numpy as np
import pygmsh, gmsh

from .simplex_mesh import SimplexMesh

"""
Mesh.Algorithm = 5   # Delaunay
Mesh.Algorithm = 6   # Frontal-Delaunay (usually nicest)
Mesh.Algorithm = 8   # Delquad
"""


# ================================================================ #
def generate(geometry_builder, h=0.2, smooth=10, **kwargs):

    with pygmsh.geo.Geometry() as geom:

        geometry_builder(geom, h=h, **kwargs)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Smoothing", smooth)

        mesh = geom.generate_mesh(
            dim=2,
            verbose=False,
        )

    return SimplexMesh(mesh)

# ================================================================ #
def unitline(h=0.5, a=0.0, b=1.0, use_pygmsh=False):
    """
    Simple 1D mesh on [a,b].
    """

    if use_pygmsh:

        with pygmsh.geo.Geometry() as geom:
            p0 = geom.add_point([a, 0, 0], mesh_size=h)
            p1 = geom.add_point([b, 0, 0], mesh_size=h)

            line = geom.add_line(p0, p1)

            geom.add_physical(p0, label="10000")
            geom.add_physical(p1, label="10001")
            geom.add_physical(line, label="1000")

            mesh = geom.generate_mesh()

        return SimplexMesh(mesh)

    # ------------------------------------------------------------ #
    # Lightweight fake meshio-like object for fast testing
    class Cell1D:
        def __init__(self, celltype, data):
            self.type = celltype
            self.data = data

    class Mesh1D:
        def __init__(self, a=0, b=1, h=0.1):
            N = int((b - a) / h + 1)

            self.points = np.stack(
                [
                    np.linspace(a, b, N),
                    np.zeros(N),
                    np.zeros(N),
                ],
                axis=1,
            )

            linedata = np.stack(
                [
                    np.arange(0, N - 1),
                    np.arange(1, N),
                ],
                axis=1,
            )

            vertexdata = np.array([[0], [N - 1]])

            self.cells = [
                Cell1D("line", linedata),
                Cell1D("vertex", vertexdata),
            ]

            self.cells_dict = {c.type: c.data for c in self.cells}

            self.cell_sets = {
                "10000": [None, np.array([0])],
                "10001": [None, np.array([1])],
                "1000": [np.arange(0, N - 1), None],
            }

    return SimplexMesh(Mesh1D(a, b, h))


# ================================================================ #
def add_unitsquare(geom, h=0.2, a=1.0):
    p = geom.add_rectangle(
        xmin=-a,
        xmax=a,
        ymin=-a,
        ymax=a,
        z=0,
        mesh_size=h,
    )

    geom.add_physical(p.surface, label="100")

    for i, line in enumerate(p.lines):
        geom.add_physical(line, label=f"{1000 + i}")


# ================================================================ #
def unitsquare(h=0.2, a=1.0):
    return generate(add_unitsquare, h=h, a=a)


# ================================================================ #
def add_unitcube(geom, h=0.5):
    x, y, z = [-1, 1], [-1, 1], [-1, 1]

    p = geom.add_rectangle(
        xmin=x[0],
        xmax=x[1],
        ymin=y[0],
        ymax=y[1],
        z=z[0],
        mesh_size=h,
    )

    geom.add_physical(p.surface, label="100")

    axis = [0, 0, z[1] - z[0]]

    top, vol, lat = geom.extrude(p.surface, axis)

    geom.add_physical(top, label="105")

    geom.add_physical(lat[0], label="101")
    geom.add_physical(lat[1], label="102")
    geom.add_physical(lat[2], label="103")
    geom.add_physical(lat[3], label="104")

    geom.add_physical(vol, label="10")


# ================================================================ #
def unitcube(h=0.5):
    return generate(add_unitcube, h=h)


# ================================================================ #
def add_backwardfacingstep(geom, h=0.5):
    X = [
        [-1.0, 1.0],
        [-1.0, 0.0],
        [0.0, 0.0],
        [0.0, -1.0],
        [3.0, -1.0],
        [3.0, 1.0],
    ]

    p = geom.add_polygon(
        points=np.insert(np.array(X), 2, 0, axis=1),
        mesh_size=h,
    )

    geom.add_physical(p.surface, label="100")

    for i, line in enumerate(p.lines):
        geom.add_physical(line, label=f"{1000 + i}")


# ================================================================ #
def backwardfacingstep(h=0.5):
    return generate(add_backwardfacingstep, h=h)


# ================================================================ #
def add_backwardfacingstep3d(geom, h=0.5):
    X = [
        [-1.0, 1.0],
        [-1.0, 0.0],
        [0.0, 0.0],
        [0.0, -1.0],
        [3.0, -1.0],
        [3.0, 1.0],
    ]

    p = geom.add_polygon(
        points=np.insert(np.array(X), 2, -1.0, axis=1),
        mesh_size=h,
    )

    geom.add_physical(p.surface, label="100")

    axis = [0, 0, 2]

    top, vol, lat = geom.extrude(p.surface, axis)

    nlat = len(lat)

    geom.add_physical(top, label=f"{101 + nlat}")

    for i in range(nlat):
        geom.add_physical(lat[i], label=f"{101 + i}")

    geom.add_physical(vol, label="10")


# ================================================================ #
def backwardfacingstep3d(h=0.5):
    return generate(add_backwardfacingstep3d, h=h)


# ================================================================ #
def add_equilateral(geom, h):
    a = 1.0

    X = [
        [-0.5 * a, 0, 0],
        [0, -0.5 * np.sqrt(3) * a, 0],
        [0.5 * a, 0, 0],
        [0, 0.5 * np.sqrt(3) * a, 0],
    ]

    p = geom.add_polygon(X, mesh_size=h)

    geom.add_physical(p.surface, label="100")

    for i, line in enumerate(p.lines):
        geom.add_physical(line, label=1000 + i)


# ================================================================ #
def equilateral(h=0.2):
    return generate(add_equilateral, h=h)