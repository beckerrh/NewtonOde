# -*- coding: utf-8 -*-
"""
Newest-vertex bisection refinement for FEM2D mesh.

Robust strategy:
    1. build child cells from explicit NVB templates,
    2. propagate boundary labels by splitting labelled boundary edges,
    3. rebuild all derived topology/geometry from scratch.

This avoids fragile incremental updates of faces_of_cells/cells_of_faces.
"""

import numpy as np
from contextlib import nullcontext

from FEM2D.mesh.mesh_edges import sorted_edge
from FEM2D.mesh.mesh_checks import check_no_degenerate_cells, check_no_nonmanifold_edges, check_boundary_normals

from collections import defaultdict, deque

from FEM2D.mesh.refinement_info import RefinementInfo
from FEM2D.mesh.simplex_mesh import SimplexMesh


def _timed(timer, name):
    return timer(name) if timer is not None else nullcontext()

def _cell_edges(tri):
    a, b, c = map(int, tri)
    return {
        sorted_edge(a, b),
        sorted_edge(b, c),
        sorted_edge(c, a),
    }


def _refine_cell_nvb_iterative(tri, refedge, marked_edges, points, new_points, edge_to_mid):
    out_cells = []
    out_refs = []

    # stack = [(list(map(int, tri)), sorted_edge(refedge[0], refedge[1]))]
    a, b = int(refedge[0]), int(refedge[1])
    stack = [(list(map(int, tri)), (a, b) if a < b else (b, a))]

    while stack:
        tri, refedge = stack.pop()

        v0, v1, v2 = tri
        # e01 = sorted_edge(v0, v1)
        # e12 = sorted_edge(v1, v2)
        # e20 = sorted_edge(v2, v0)
        e01 = (v0, v1) if v0 < v1 else (v1, v0)
        e12 = (v1, v2) if v1 < v2 else (v2, v1)
        e20 = (v2, v0) if v2 < v0 else (v0, v2)

        if e01 not in marked_edges and e12 not in marked_edges and e20 not in marked_edges:
            out_cells.append(tri)
            out_refs.append(refedge)
            continue

        a, b = refedge
        c = _third_vertex(tri, a, b)

        m = _edge_midpoint((a, b), points, edge_to_mid, new_points)

        child0 = [c, a, m]
        child1 = [b, c, m]

        # ref0 = sorted_edge(c, a)
        # ref1 = sorted_edge(b, c)
        ref0 = (c, a) if c < a else (a, c)
        ref1 = (b, c) if b < c else (c, b)

        stack.append((child1, ref1))
        stack.append((child0, ref0))

    return out_cells, out_refs
def _refine_cell_recursive_nvb(tri, refedge, marked_edges, points, new_points, edge_to_mid,):
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
    for icell, tri in enumerate(mesh.topology.cells):
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
            raise RuntimeError("mesh.refedges is not aligned with mesh.topology.cells")

def _longest_edge_refedges(mesh):
    refedges = []
    for tri in mesh.topology.cells:
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

def _edge_midpoint(edge, points, edge_to_mid, new_points):
    a, b = edge
    e = (a, b) if a < b else (b, a)
    return edge_to_mid[e]

# ====================================================================== #
def _split_boundary_labels(mesh, edge_to_mid):
    """
    Split old labelled boundary edges into child edges.
    """
    new_bdrylabels = {}

    for label, faces in mesh.labels.boundary.items():
        new_faces = []

        for f in faces:
            a, b = mesh.topology.faces[f]
            # e = sorted_edge(a, b)
            # a, b = int(a), int(b)
            e = (a, b) if a < b else (b, a)

            if e in edge_to_mid:
                m = edge_to_mid[e]
                # new_faces.append(sorted_edge(a, m))
                # new_faces.append(sorted_edge(m, b))
                new_faces.append((a, m) if a < m else (m, a))
                new_faces.append((m, b) if m < b else (b, m))
            else:
                new_faces.append(e)

        new_bdrylabels[label] = new_faces

    return new_bdrylabels

# ====================================================================== #
def _apply_boundary_labels(mesh, boundary_edge_labels):
    """
    Convert boundary edge tuples to rebuilt face indices.
    """
    if not boundary_edge_labels:
        mesh.labels.boundary = {}
        return

    face_to_id = mesh.edge2face
    # face_to_id = face_dict(mesh.topology.faces)
    bdrylabels = {}

    for label, edges in boundary_edge_labels.items():
        ids = []

        for a, b in edges:
            edge = sorted_edge(a, b)
            iface = face_to_id.get(edge)

            if iface is not None:
                ids.append(iface)

        bdrylabels[label] = np.asarray(ids, dtype=int)

    mesh.labels.boundary = bdrylabels

# ====================================================================== #
def refine_nvb(mesh, marked, debug=False, timer=None):
    """
    Variant of refine_nvb with preallocated global cell/refedge/label arrays.
    Intended for timing comparison against refine_nvb.
    """
    marked = np.asarray(marked, dtype=bool)

    if marked.ndim != 1:
        raise ValueError("marked must be a 1D boolean array")

    ncells = mesh.topology.cells.shape[0]

    if marked.shape[0] != ncells:
        raise ValueError("wrong size for marked")

    faces = mesh.topology.faces
    fmap = mesh.edge2face
    cells = mesh.topology.cells
    points = mesh.points

    # ------------------------------------------------------------------ #
    # Closure propagation
    # ------------------------------------------------------------------ #
    with _timed(timer, "closure"):
        refine_edge = np.zeros(mesh.topology.faces.shape[0], dtype=bool)

        refedges = _ensure_refedges(mesh)

        if debug:
            check_refedges(mesh, "entry")

        cell_ref_faces = np.empty(ncells, dtype=int)

        for icell in range(ncells):
            a, b = map(int, refedges[icell])
            edge = (a, b) if a < b else (b, a)
            cell_ref_faces[icell] = fmap[edge]

        queue = deque()

        for icell in np.flatnonzero(marked):
            iref = cell_ref_faces[icell]
            if not refine_edge[iref]:
                refine_edge[iref] = True
                queue.append(iref)

        while queue:
            iface = queue.popleft()

            for icell in mesh.topology.cells_of_faces[iface]:
                if icell < 0:
                    continue

                iref = cell_ref_faces[icell]
                if not refine_edge[iref]:
                    refine_edge[iref] = True
                    queue.append(iref)

    # ------------------------------------------------------------------ #
    # New midpoint vertices
    # ------------------------------------------------------------------ #
    with _timed(timer, "new_points"):
        edge_to_mid = {}

        refined_faces = np.flatnonzero(refine_edge)
        n_old = points.shape[0]
        n_new = refined_faces.size

        new_points = np.empty((n_old + n_new, points.shape[1]), dtype=points.dtype)
        new_points[:n_old] = points

        for j, iface in enumerate(refined_faces):
            a, b = map(int, faces[iface])
            edge = (a, b) if a < b else (b, a)

            imid = n_old + j
            edge_to_mid[edge] = imid
            new_points[imid] = 0.5 * (points[a] + points[b])

    # ------------------------------------------------------------------ #
    # Old cell labels
    # ------------------------------------------------------------------ #
    with _timed(timer, "old_celllabels"):
        old_celllabels = np.empty(ncells, dtype=int)

        for label, ids in mesh.labels.cell.items():
            old_celllabels[np.asarray(ids, dtype=int)] = label

    # ------------------------------------------------------------------ #
    # Marked edges as tuples
    # ------------------------------------------------------------------ #
    with _timed(timer, "marked_edges_set"):
        marked_edges = set()

        for iface in refined_faces:
            a, b = map(int, faces[iface])
            marked_edges.add((a, b) if a < b else (b, a))

    # ------------------------------------------------------------------ #
    # Refine cells, with preallocated global arrays
    # ------------------------------------------------------------------ #
    with _timed(timer, "refine_cells_loop"):
        old_npoints = points.shape[0]
        old_ncells = cells.shape[0]

        # In conforming 2D NVB closure, 4*ncells is a safe first upper bound.
        # If this ever fails, increase to 8*ncells or add dynamic growth.
        max_new_cells = 4 * ncells

        new_cells = np.empty((max_new_cells, 3), dtype=np.int64)
        new_refedges = np.empty((max_new_cells, 2), dtype=np.int64)
        new_celllabels = np.empty(max_new_cells, dtype=np.int64)
        parent_cell_of_child = np.empty(max_new_cells, dtype=np.int64)

        nnew = 0
        child_cells_of_parent = {}

        midpoint_parents = {
            int(mid): tuple(map(int, edge))
            for edge, mid in edge_to_mid.items()
        }

        for icell in range(ncells):
            tri = list(map(int, cells[icell]))
            refedge = tuple(map(int, refedges[icell]))

            childs, child_refs = _refine_cell_nvb_iterative(
                tri,
                refedge,
                marked_edges,
                points,
                new_points,
                edge_to_mid,
            )

            first_child = nnew
            nchild = len(childs)

            if nnew + nchild > max_new_cells:
                raise RuntimeError(
                    f"max_new_cells too small: need at least {nnew + nchild}, "
                    f"allocated {max_new_cells}"
                )

            label = old_celllabels[icell]

            for child, ref in zip(childs, child_refs):
                new_cells[nnew, :] = child
                new_refedges[nnew, :] = ref
                new_celllabels[nnew] = label
                parent_cell_of_child[nnew] = icell
                nnew += 1

            child_cells_of_parent[icell] = list(range(first_child, nnew))

        new_cells = new_cells[:nnew]
        new_refedges = new_refedges[:nnew]
        new_celllabels = new_celllabels[:nnew]
        parent_cell_of_child = parent_cell_of_child[:nnew]

    # ------------------------------------------------------------------ #
    # Build new mesh
    # ------------------------------------------------------------------ #
    with _timed(timer, "construct_new_mesh"):
        boundary_edge_labels = _split_boundary_labels(mesh, edge_to_mid)

        labels = {
            "names": mesh.labels.names.copy(),
        }

        mesh2 = SimplexMesh(
            points=np.asarray(new_points, dtype=float),
            cells=new_cells,
            labels=labels,
            rebuild=False,
            check=False,
        )

        mesh2.refedges = new_refedges
        mesh2.cell_markers = new_celllabels

    # ------------------------------------------------------------------ #
    # Rebuild topology/geometry
    # ------------------------------------------------------------------ #
    with _timed(timer, "finalize"):
        mesh2.finalize_after_topology_change(timer=timer)

    with _timed(timer, "bdry_labels"):
        _apply_boundary_labels(mesh2, boundary_edge_labels)

    if debug:
        check_refedges(mesh2, "after finalize")
        check_no_degenerate_cells(mesh2.cells)
        debug_nonmanifold_edges(mesh2.cells)
        check_no_nonmanifold_edges(mesh2.cells)
        check_boundary_normals(mesh2)

    info = RefinementInfo(
        old_npoints=old_npoints,
        new_npoints=mesh2.points.shape[0],
        old_ncells=old_ncells,
        new_ncells=mesh2.topology.cells.shape[0],
        midpoint_parents=midpoint_parents,
        child_cells_of_parent=child_cells_of_parent,
        parent_cell_of_child=parent_cell_of_child,
    )

    return mesh2, info# ====================================================================== #
if __name__ == "__main__":

    import cProfile
    import pstats
    import matplotlib.pyplot as plt

    from FEM2D.mesh import testmeshes

    mesh = testmeshes.unitsquare(h=0.2)

    def run(mesh):
        for k in range(12):

            xc = mesh.cell_centers[:, 0]
            yc = mesh.cell_centers[:, 1]

            marked = xc**2 + yc**2 < 0.35**2

            mesh, info = refine_nvb(mesh, marked)
            print(
                f"iter={k:2d} "
                f"npoints={mesh.points.shape[0]:6d} "
                f"ncells={mesh.topology.cells.shape[0]:6d}"
            )

    cProfile.run("run(mesh)", "nvb.prof")

    stats = pstats.Stats("nvb.prof")
    stats.sort_stats("cumtime").print_stats(40)

    mesh.plot_boundary()
    plt.gca().set_aspect("equal")
    plt.show()