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

from FEM2D.mesh.mesh_checks import check_no_degenerate_cells, check_no_nonmanifold_edges, check_boundary_normals


from FEM2D.mesh.refinement_info import RefinementInfo
from FEM2D.mesh.simplex_mesh import SimplexMesh
from FEM2D.mesh.refine_cell_nvb_iterative import _refine_cell_nvb_iterative, _refine_cell_nvb_iterative_keyed


def _timed(timer, name):
    return timer(name) if timer is not None else nullcontext()


def _longest_edge_refedges(mesh):
    refedges = []
    points = mesh.geometry.points

    for tri in mesh.topology.cells:
        a, b, c = map(int, tri)

        edges = ((a, b), (b, c), (c, a))
        lengths = (
            np.linalg.norm(points[a] - points[b]),
            np.linalg.norm(points[b] - points[c]),
            np.linalg.norm(points[c] - points[a]),
        )

        i, j = edges[int(np.argmax(lengths))]
        refedges.append((i, j) if i < j else (j, i))

    return np.asarray(refedges, dtype=np.int64)

def _ensure_refedges(mesh):
    if not hasattr(mesh, "refedges") or mesh.refedges is None:
        mesh.refedges = _longest_edge_refedges(mesh)
    return mesh.refedges



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
            e = (a, b) if a < b else (b, a)

            if e in edge_to_mid:
                m = edge_to_mid[e]
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
    bdrylabels = {}

    for label, edges in boundary_edge_labels.items():
        ids = []

        for a, b in edges:
            edge = (a, b) if a < b else (b, a)
            iface = face_to_id.get(edge)

            if iface is not None:
                ids.append(iface)

        bdrylabels[label] = np.asarray(ids, dtype=int)

    mesh.labels.boundary = bdrylabels

def _refine_cell_nvb_write(tri, refedge,
    marked_keys, key_to_mid, nkey,
    new_cells, new_refedges,
    pos, maxpos):
    a = int(tri[0]); b = int(tri[1]); c = int(tri[2])
    r0 = int(refedge[0]); r1 = int(refedge[1])

    stack = [(a, b, c, r0, r1)]

    while stack:
        a, b, c, r0, r1 = stack.pop()

        key = r0 * nkey + r1 if r0 < r1 else r1 * nkey + r0

        if key not in marked_keys:
            if pos >= maxpos:
                raise RuntimeError(f"max_new_cells too small: allocated {maxpos}")

            new_cells[pos, 0] = a
            new_cells[pos, 1] = b
            new_cells[pos, 2] = c
            new_refedges[pos, 0] = r0
            new_refedges[pos, 1] = r1
            pos += 1
            continue

        m = key_to_mid[key]

        if a != r0 and a != r1:
            z = a
        elif b != r0 and b != r1:
            z = b
        else:
            z = c

        stack.append((z, m, r1, z, r1))
        stack.append((z, r0, m, z, r0))

    return pos
def close_marked_faces_nvb(cells_of_faces, cell_ref_faces, marked):
    """
    NVB closure using face indices.

    Parameters
    ----------
    cells_of_faces : (nfaces, 2) int array
        Adjacent cells of each face, with -1 on boundary.

    cell_ref_faces : (ncells,) int array
        Global face index of the refinement edge of each cell.

    marked : (ncells,) bool array
        Initially marked cells.

    Returns
    -------
    marked_bool : (ncells,) bool array
        Closed set of marked cells.

    refine_face : (nfaces,) bool array
        Faces/edges that must be bisected.

    refined_faces : int array
        Indices of faces/edges that must be bisected.
    """
    marked_bool = np.asarray(marked, dtype=bool).copy()

    ncells = cell_ref_faces.shape[0]
    nfaces = cells_of_faces.shape[0]

    refine_face = np.zeros(nfaces, dtype=bool)

    queue = np.empty(ncells, dtype=np.int64)
    initial = np.flatnonzero(marked_bool)

    queue[:initial.size] = initial
    head = 0
    tail = initial.size

    while head < tail:
        icell = queue[head]
        head += 1

        iface = cell_ref_faces[icell]
        refine_face[iface] = True

        c0, c1 = cells_of_faces[iface]
        nb = c1 if c0 == icell else c0

        if nb >= 0 and not marked_bool[nb]:
            marked_bool[nb] = True
            queue[tail] = nb
            tail += 1

    refined_faces = np.flatnonzero(refine_face)

    return marked_bool, refine_face, refined_faces
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
    points = mesh.geometry.points

    # ------------------------------------------------------------------ #
    # Closure propagation
    # ------------------------------------------------------------------ #
    with _timed(timer, "closure"):
        refedges = _ensure_refedges(mesh)

        if debug:
            if debug:
                from FEM2D.mesh.nvb_debug import check_refedges, debug_nonmanifold_edges
            check_refedges(mesh, "entry")

        cell_ref_faces = np.empty(ncells, dtype=np.int64)

        for icell in range(ncells):
            a, b = map(int, refedges[icell])
            edge = (a, b) if a < b else (b, a)
            cell_ref_faces[icell] = mesh.edge2face[edge]

        marked_bool, refine_edge, refined_faces = close_marked_faces_nvb(
            mesh.topology.cells_of_faces,
            cell_ref_faces,
            marked,
        )
    # ------------------------------------------------------------------ #
    # Marked edges as tuples
    # ------------------------------------------------------------------ #
    with _timed(timer, "marked_edges_set"):
        marked_edges = set()

        for iface in refined_faces:
            a, b = map(int, faces[iface])
            marked_edges.add((a, b) if a < b else (b, a))

    # ------------------------------------------------------------------ #
    # New midpoint vertices
    # ------------------------------------------------------------------ #
    with _timed(timer, "new_points"):
        edge_to_mid = {}

        # refined_faces = np.flatnonzero(refine_edge)
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
    # Refine cells, with preallocated global arrays
    # ------------------------------------------------------------------ #
    with _timed(timer, "refine_cells_loop"):

        nkey = len(new_points)
        marked_edge_keys = set()
        for a, b in marked_edges:
            if a < b:
                marked_edge_keys.add(a * nkey + b)
            else:
                marked_edge_keys.add(b * nkey + a)

        edgekey_to_mid = {}
        for (a, b), m in edge_to_mid.items():
            if a < b:
                edgekey_to_mid[a * nkey + b] = m
            else:
                edgekey_to_mid[b * nkey + a] = m

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

            first_child = nnew

            nnew = _refine_cell_nvb_write(
                cells[icell],
                refedges[icell],
                marked_edge_keys,
                edgekey_to_mid,
                nkey,
                new_cells,
                new_refedges,
                nnew,
                max_new_cells,
            )

            if nnew > max_new_cells:
                raise RuntimeError(
                    f"max_new_cells too small: need at least {nnew}, "
                    f"allocated {max_new_cells}"
                )

            new_celllabels[first_child:nnew] = old_celllabels[icell]
            parent_cell_of_child[first_child:nnew] = icell

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
        new_npoints=mesh2.geometry.points.shape[0],
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

            xc = mesh.geometry.cell_centers[:, 0]
            yc = mesh.geometry.cell_centers[:, 1]

            marked = xc**2 + yc**2 < 0.35**2

            mesh, info = refine_nvb(mesh, marked)
            print(
                f"iter={k:2d} "
                f"npoints={mesh.geometry.points.shape[0]:6d} "
                f"ncells={mesh.topology.cells.shape[0]:6d}"
            )

    cProfile.run("run(mesh)", "nvb.prof")

    stats = pstats.Stats("nvb.prof")
    stats.sort_stats("cumtime").print_stats(40)

    mesh.plot_boundary()
    plt.gca().set_aspect("equal")
    plt.show()