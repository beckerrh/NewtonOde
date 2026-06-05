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
def add_unitsquare(geom, h=0.2, a=1.0, boundary_projectors=None):
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
def add_unitcube(geom, h=0.5, boundary_projectors=None):
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
def add_backwardfacingstep(geom, h=0.5, boundary_projectors=None):
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
def add_backwardfacingstep3d(geom, h=0.5, boundary_projectors=None):
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
def add_equilateral(geom, h, boundary_projectors=None):
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

