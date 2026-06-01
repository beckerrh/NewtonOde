# -*- coding: utf-8 -*-
"""Debug checks for triangular meshes."""
import numpy as np
from collections import Counter

def check_faces_of_cells_local_order(mesh):
    def edge(a, b):
        return (a, b) if a < b else (b, a)

    for ic, tri in enumerate(mesh.topology.cells):
        v0, v1, v2 = map(int, tri)

        expected = [
            edge(v1, v2),
            edge(v2, v0),
            edge(v0, v1),
        ]

        for iloc in range(3):
            f = mesh.topology.faces_of_cells[ic, iloc]
            got = edge(*mesh.topology.faces[f])

            if got != expected[iloc]:
                raise ValueError(
                    f"bad faces_of_cells ordering at cell {ic}, local {iloc}: "
                    f"got {got}, expected {expected[iloc]}"
                )
def check_boundary_normals(mesh, tol=1e-12):
    pc = mesh.geometry.cell_centers[:, :mesh.dimension]
    pf = mesh.geometry.face_centers[:, :mesh.dimension]
    normals = mesh.geometry.normals[:, :mesh.dimension]

    bad = []

    for color, faces in mesh.labels.boundary.items():
        for f in faces:
            cells = mesh.topology.cells_of_faces[f]
            cells = np.asarray(cells)
            cells = cells[cells >= 0]
            if len(cells) != 1:
                bad.append((color, f, "not one boundary cell", cells))
                continue

            ic = cells[0]
            # outward normal should point from cell center to face center
            # hence dot(n_f, pf - pc) should be positive
            s = np.dot(normals[f], pf[f] - pc[ic])
            if s <= tol:
                bad.append((color, f, s))

    if bad:
        print("BAD BOUNDARY NORMALS:", bad[:20])
        raise ValueError(f"{len(bad)} boundary faces have non-outward normals")

def check_no_degenerate_cells(cells):
    for icell, tri in enumerate(cells):
        if len(set(map(int, tri))) != 3:
            raise ValueError(f"Degenerate triangle {icell}: {tri}")



def check_no_nonmanifold_edges(cells):
    counter = Counter()

    for tri in cells:
        a, b, c = map(int, tri)

        for i, j in ((a, b), (b, c), (c, a)):
            edge = (i, j) if i < j else (j, i)
            counter[edge] += 1

    for edge, count in counter.items():
        if count > 2:
            raise ValueError(f"Nonmanifold edge {edge}: used by {count} cells")

def check_mesh(mesh):
    cells = getattr(mesh, "cells", getattr(mesh, "cells", None))
    if cells is None:
        raise AttributeError("mesh has neither 'cells' nor 'cells'")
    check_no_degenerate_cells(mesh.topology.cells)
    check_no_nonmanifold_edges(mesh.topology.cells)
    check_faces_of_cells_local_order(mesh)
    print("nnodes", mesh.nnodes, mesh.geometry.points.shape[0])
    print("ncells", mesh.ncells, mesh.cells.shape[0])
    print("nfaces", mesh.nfaces, mesh.topology.faces.shape[0])
    print("simp max", mesh.cells.max())
    print("faces max", mesh.topology.faces.max())
    print("dV min/max", mesh.geometry.cell_volumes.min(), mesh.geometry.cell_volumes.max())
    print("bad dV", np.sum(mesh.geometry.cell_volumes <= 0))
    print("cell labels", {k: len(v) for k, v in mesh.labels.cell.items()})
    print("bdry labels", {k: len(v) for k, v in mesh.labels.boundary.items()})

