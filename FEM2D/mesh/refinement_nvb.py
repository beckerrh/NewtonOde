# -*- coding: utf-8 -*-
"""
Newest-vertex bisection refinement for FEM2D mesh.

Robust strategy:
    1. build child cells from explicit NVB templates,
    2. propagate boundary labels by splitting labelled boundary edges,
    3. rebuild all derived topology/geometry from scratch.

This avoids fragile incremental updates of facesOfCells/cellsOfFaces.
"""

import numpy as np

from FEM2D.mesh.mesh_edges import compute_edges, face_dict, sorted_edge
from FEM2D.mesh.mesh_checks import check_no_degenerate_cells, check_no_nonmanifold_edges, check_boundary_normals

from collections import defaultdict

def _cell_edges(tri):
    a, b, c = map(int, tri)
    return {
        sorted_edge(a, b),
        sorted_edge(b, c),
        sorted_edge(c, a),
    }


def _refine_cell_recursive_nvb(tri, refedge, marked_edges, points, new_points, edge_to_mid):
    tri = list(map(int, tri))
    refedge = sorted_edge(refedge[0], refedge[1])

    # stop if this triangle has no marked boundary edge
    if _cell_edges(tri).isdisjoint(marked_edges):
        return [tri], [refedge]

    a, b = refedge
    c = _third_vertex(tri, a, b)

    m = _edge_midpoint(
        (a, b),
        points,
        edge_to_mid,
        new_points,
    )

    child0 = [c, a, m]
    child1 = [b, c, m]

    ref0 = sorted_edge(c, a)
    ref1 = sorted_edge(b, c)

    cells0, refs0 = _refine_cell_recursive_nvb(
        child0, ref0, marked_edges, points, new_points, edge_to_mid
    )
    cells1, refs1 = _refine_cell_recursive_nvb(
        child1, ref1, marked_edges, points, new_points, edge_to_mid
    )

    return cells0 + cells1, refs0 + refs1
def check_refedges(mesh, where=""):
    for icell, tri in enumerate(mesh.cells):
        tri = list(map(int, tri))
        e = tuple(map(int, mesh.refedges[icell]))

        cell_edges = {
            sorted_edge(tri[0], tri[1]),
            sorted_edge(tri[1], tri[2]),
            sorted_edge(tri[2], tri[0]),
        }

        if sorted_edge(e[0], e[1]) not in cell_edges:
            print("BAD REFEDGE", where)
            print("  icell =", icell)
            print("  tri   =", tri)
            print("  ref   =", e)
            print("  cell_edges =", cell_edges)
            raise RuntimeError("mesh.refedges is not aligned with mesh.cells")

def _longest_edge_refedges(mesh):
    refedges = []
    for tri in mesh.cells:
        a, b, c = map(int, tri)
        edges = [(a, b), (b, c), (c, a)]
        lengths = [
            np.linalg.norm(mesh.points[i] - mesh.points[j])
            for i, j in edges
        ]
        e = edges[int(np.argmax(lengths))]
        refedges.append(sorted_edge(e[0], e[1]))
    return np.asarray(refedges, dtype=int)


def _ensure_refedges(mesh):
    if not hasattr(mesh, "refedges") or mesh.refedges is None:
        mesh.refedges = _longest_edge_refedges(mesh)
    return mesh.refedges


# def _face_index_from_edge(mesh, edge):
#     fmap = face_dict(mesh.faces)
#     return fmap[sorted_edge(edge[0], edge[1])]
# def ensure_midpoint(edge, points, new_points, edge_to_mid):
#     edge = tuple(sorted(map(int, edge)))
#     if edge not in edge_to_mid:
#         a, b = edge
#         edge_to_mid[edge] = len(new_points)
#         new_points.append(0.5 * (points[a] + points[b]))
#     return edge_to_mid[edge]
def debug_nonmanifold_edges(cells):
    edge_to_cells = defaultdict(list)
    for ic, c in enumerate(cells):
        c = list(map(int, c))
        for a, b in [(c[0], c[1]), (c[1], c[2]), (c[2], c[0])]:
            e = tuple(sorted((a, b)))
            edge_to_cells[e].append(ic)

    for e, cs in edge_to_cells.items():
        if len(cs) > 2:
            print("NONMANIFOLD EDGE", e, "cells", cs)
            for ic in cs:
                print("  cell", ic, cells[ic])
            raise ValueError(f"Nonmanifold edge {e}: used by {len(cs)} cells")

def _third_vertex(tri, a, b):
    for v in tri:
        if v != a and v != b:
            return v
    raise RuntimeError("degenerate triangle")


# def _split_triangle_by_edges(tri, refined_edges, edge_to_mid):
#     """
#     Convention-free triangle splitting.
#
#     refined_edges: list of edge tuples (a,b) actually refined.
#     """
#     tri = list(map(int, tri))
#     refined_edges = [tuple(map(int, e)) for e in refined_edges]
#
#     if len(refined_edges) == 0:
#         return [tri]
#
#     if len(refined_edges) == 1:
#         a, b = refined_edges[0]
#         m = edge_to_mid[sorted_edge(a, b)]
#         c = _third_vertex(tri, a, b)
#         return [
#             [a, m, c],
#             [m, b, c],
#         ]
#
#     if len(refined_edges) == 2:
#         e0, e1 = refined_edges
#         common = set(e0).intersection(e1)
#
#         if len(common) != 1:
#             raise RuntimeError("two refined edges do not share a vertex")
#
#         a = common.pop()
#         b = e0[0] if e0[1] == a else e0[1]
#         c = e1[0] if e1[1] == a else e1[1]
#
#         mab = edge_to_mid[sorted_edge(a, b)]
#         mac = edge_to_mid[sorted_edge(a, c)]
#
#         return [
#             [a, mab, mac],
#             [mab, b, c],
#             [mab, c, mac],
#         ]
#
#     if len(refined_edges) == 3:
#         a, b, c = tri
#
#         mab = edge_to_mid[sorted_edge(a, b)]
#         mbc = edge_to_mid[sorted_edge(b, c)]
#         mca = edge_to_mid[sorted_edge(c, a)]
#
#         return [
#             [a, mab, mca],
#             [mab, b, mbc],
#             [mca, mbc, c],
#             [mab, mbc, mca],
#         ]
#
#     raise RuntimeError("too many refined edges")
# ====================================================================== #
def _edge_midpoint(edge, points, edge_to_mid, new_points):
    a, b = edge
    e = sorted_edge(a, b)

    if e in edge_to_mid:
        return edge_to_mid[e]

    pmid = 0.5 * (points[a] + points[b])

    idx = len(new_points)
    new_points.append(pmid)

    edge_to_mid[e] = idx
    return idx

# ====================================================================== #
def _split_boundary_labels(mesh,edge_to_mid):
    """
    Split old labelled boundary edges into child edges.
    """
    new_bdrylabels = {}

    if not hasattr(mesh, "bdrylabels"):
        return new_bdrylabels

    for label, faces in mesh.bdrylabels.items():

        new_faces = []

        for f in faces:

            a, b = mesh.faces[f]
            e = sorted_edge(a, b)

            if e in edge_to_mid:
                m = edge_to_mid[e]
                new_faces.append(sorted_edge(a, m))
                new_faces.append(sorted_edge(m, b))
            else:
                new_faces.append(e)

        new_bdrylabels[label] = new_faces

    return new_bdrylabels


# ====================================================================== #
def _apply_boundary_labels(mesh, boundary_edge_labels):
    """
    Convert edge tuples to rebuilt face indices.
    """
    if not boundary_edge_labels:
        mesh.bdrylabels = {}
        return

    fmap = face_dict(mesh.faces)

    bdrylabels = {}

    for label, edges in boundary_edge_labels.items():

        ids = []

        for e in edges:
            e = sorted_edge(e[0], e[1])

            if e in fmap:
                ids.append(fmap[e])

        bdrylabels[label] = np.asarray(ids, dtype=int)

    mesh.bdrylabels = bdrylabels


# ====================================================================== #
def refine_nvb(mesh, marked, method="NVB", debug=False):
    """
    Refine marked triangles using newest-vertex bisection closure.

    Parameters
    ----------
    mesh:
        SimplexMesh

    marked:
        boolean array of length ncells

    method:
        accepted for backward compatibility; currently ignored

    debug:
        if True, run consistency checks
    """
    marked = np.asarray(marked, dtype=bool)

    if marked.ndim != 1:
        raise ValueError("marked must be a 1D boolean array")

    ncells = mesh.cells.shape[0]

    if marked.shape[0] != ncells:
        raise ValueError("wrong size for marked")

    # ------------------------------------------------------------------ #
    # Edge data
    # ------------------------------------------------------------------ #
    faces_of_cells = mesh.facesOfCells
    faces = mesh.faces
    fmap = face_dict(mesh.faces)
    cells = mesh.cells
    points = mesh.points

    # ------------------------------------------------------------------ #
    # Closure propagation
    # ------------------------------------------------------------------ #
    refine_edge = np.zeros(mesh.faces.shape[0], dtype=bool)

    refedges = _ensure_refedges(mesh)
    if debug:
        check_refedges(mesh, "entry")

    for icell in np.flatnonzero(marked):
        iface = fmap[sorted_edge(refedges[icell, 0], refedges[icell, 1])]
        # iface = _face_index_from_edge(mesh, refedges[icell])
        refine_edge[iface] = True

    changed = True

    while changed:
        changed = False

        for icell in range(ncells):

            tri = list(map(int, cells[icell]))
            refedge = sorted_edge(refedges[icell, 0], refedges[icell, 1])
            # iref = _face_index_from_edge(mesh, refedge)
            iref = fmap[refedge]

            local_faces = faces_of_cells[icell]

            # If any edge of this cell is marked, the cell must be bisected
            # along its own reference edge.
            if np.any(refine_edge[local_faces]) and not refine_edge[iref]:
                refine_edge[iref] = True
                changed = True

    # ------------------------------------------------------------------ #
    edge_to_mid = {}

    new_points = [p.copy() for p in points]

    for iface, flag in enumerate(refine_edge):

        if not flag:
            continue

        a, b = faces[iface]

        _edge_midpoint(
            (a, b),
            points,
            edge_to_mid,
            new_points,
        )

    # ------------------------------------------------------------------ #
    # Build refined cells
    # ------------------------------------------------------------------ #
    new_cells = []
    new_refedges = []
    new_celllabels = []

    old_celllabels = np.empty(ncells, dtype=int)
    for label, ids in mesh.cellsoflabel.items():
        old_celllabels[np.asarray(ids, dtype=int)] = label

    marked_edges = {
        sorted_edge(faces[iface, 0], faces[iface, 1])
        for iface, flag in enumerate(refine_edge)
        if flag
    }

    for icell in range(ncells):
        tri = list(map(int, cells[icell]))
        refedge = tuple(map(int, refedges[icell]))

        childs, child_refs = _refine_cell_recursive_nvb(
            tri,
            refedge,
            marked_edges,
            points,
            new_points,
            edge_to_mid,
        )

        new_cells.extend(childs)
        new_refedges.extend(child_refs)
        new_celllabels.extend([old_celllabels[icell]] * len(childs))
    # ------------------------------------------------------------------ #

    boundary_edge_labels = _split_boundary_labels(mesh, edge_to_mid)

    mesh.points = np.asarray(new_points)

    mesh.cells = np.asarray(new_cells, dtype=int)
    mesh.celllabels = np.asarray(new_celllabels, dtype=int)
    mesh.refedges = np.asarray(new_refedges, dtype=int)
    if debug:
        check_refedges(mesh, "before finalize")
    # print("len new_cells", len(new_cells))
    # print("len new_celllabels", len(new_celllabels))
    # print("unique celllabels", np.unique(mesh.celllabels, return_counts=True))

    # ------------------------------------------------------------------ #
    # Rebuild topology
    # ------------------------------------------------------------------ #
    mesh.finalize_after_topology_change()
    if debug:
        check_refedges(mesh, "after finalize")
    # ------------------------------------------------------------------ #
    # Boundary labels
    # ------------------------------------------------------------------ #

    _apply_boundary_labels(
        mesh,
        boundary_edge_labels,
    )

    # ------------------------------------------------------------------ #
    # Safety checks
    # ------------------------------------------------------------------ #
    if debug:
        check_no_degenerate_cells(mesh.cells)
        debug_nonmanifold_edges(mesh.cells)
        check_no_nonmanifold_edges(mesh.cells)
        check_boundary_normals(mesh)

    return mesh


# ====================================================================== #
if __name__ == "__main__":

    import cProfile
    import pstats
    import matplotlib.pyplot as plt

    from FEM2D.mesh import testmeshes

    mesh = testmeshes.unitsquare(h=0.2)

    def run():
        for k in range(6):

            xc = mesh.pointsc[:, 0]
            yc = mesh.pointsc[:, 1]

            marked = xc**2 + yc**2 < 0.35**2

            refine_nvb(mesh, marked)

            print(
                f"iter={k:2d} "
                f"npoints={mesh.points.shape[0]:6d} "
                f"ncells={mesh.cells.shape[0]:6d}"
            )

    cProfile.run("run()", "nvb.prof")

    stats = pstats.Stats("nvb.prof")
    stats.sort_stats("cumtime").print_stats(40)

    mesh.plot(bdry=True)
    plt.gca().set_aspect("equal")
    plt.show()