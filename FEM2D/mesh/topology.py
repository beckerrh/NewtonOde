import numpy as np


def construct_inner_faces(mesh):
    mesh.topology.inner_faces = mesh.topology.cells_of_faces[:, 1] >= 0
    mesh.topology.cells_of_inner_faces = mesh.topology.cells_of_faces[mesh.topology.inner_faces]


def construct_faces_from_cells(mesh, build_edge2face=True):
    """
    Vectorized construction of triangular mesh topology.

    Sets
    ----
    mesh.topology.faces          : (nfaces, 2)
    mesh.topology.faces_of_cells : (ncells, 3)
    mesh.topology.cells_of_faces : (nfaces, 2)
    mesh.edge2face      : dict[(a,b)] -> iface, optional

    Local face convention:
        local face 0 = edge (1,2)
        local face 1 = edge (2,0)
        local face 2 = edge (0,1)
    """
    import numpy as np

    cells = np.asarray(mesh.topology.cells, dtype=np.int64)
    ncells = cells.shape[0]

    # ------------------------------------------------------------
    # Build all local edges.
    # ------------------------------------------------------------
    edges = np.empty((3 * ncells, 2), dtype=np.int64)

    edges[0::3, :] = cells[:, [1, 2]]
    edges[1::3, :] = cells[:, [2, 0]]
    edges[2::3, :] = cells[:, [0, 1]]

    # Sort the two vertices of each edge.
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    edges[:, 0] = lo
    edges[:, 1] = hi

    npoints = mesh.geometry.points.shape[0]

    keys = lo * npoints + hi
    order = np.argsort(keys)
    #
    # keys_s = keys[order]
    # edges_s = edges[order]
    owners_s = order // 3
    local_s = order % 3

    keys_s = keys[order]

    is_new = np.empty(keys_s.shape[0], dtype=bool)
    is_new[0] = True
    is_new[1:] = keys_s[1:] != keys_s[:-1]

    face_ids_s = np.cumsum(is_new, dtype=np.int64) - 1
    first_pos = np.flatnonzero(is_new)
    unique_keys = keys_s[first_pos]

    faces = np.column_stack((unique_keys // npoints, unique_keys % npoints)).astype(np.int64)

    # ------------------------------------------------------------
    # faces_of_cells[icell, iloc] = global face index
    # ------------------------------------------------------------
    faces_of_cells = np.empty((ncells, 3), dtype=np.int64)
    faces_of_cells[owners_s, local_s] = face_ids_s

    # ------------------------------------------------------------
    # cells_of_faces[iface] = [left_cell, right_cell or -1]
    # ------------------------------------------------------------
    first_pos = np.flatnonzero(is_new)
    # counts = np.diff(np.r_[first_pos, edges_s.shape[0]])
    counts = np.diff(np.r_[first_pos, keys_s.shape[0]])

    if np.any(counts > 2):
        bad = int(np.flatnonzero(counts > 2)[0])
        raise ValueError(
            f"non-manifold edge {tuple(faces[bad])}: "
            f"used by {int(counts[bad])} cells"
        )

    nfaces = int(face_ids_s[-1]) + 1
    cells_of_faces = np.full((nfaces, 2), -1, dtype=np.int64)
    cells_of_faces[:, 0] = owners_s[first_pos]

    has_second = counts == 2
    second_pos = first_pos[has_second] + 1
    cells_of_faces[has_second, 1] = owners_s[second_pos]

    # ------------------------------------------------------------
    # Install on mesh.
    # ------------------------------------------------------------
    mesh.topology.faces = faces
    mesh.topology.faces_of_cells = faces_of_cells
    mesh.topology.cells_of_faces = cells_of_faces
    mesh.nfaces = nfaces

    if build_edge2face:
        mesh.edge2face = {
            (int(a), int(b)): int(i)
            for i, (a, b) in enumerate(faces)
        }
    else:
        mesh.edge2face = None

    # ------------------------------------------------------------
    # Cheap invariants.
    # ------------------------------------------------------------
    assert mesh.topology.faces.shape == (mesh.nfaces, 2)
    assert mesh.topology.faces_of_cells.shape == (ncells, 3)
    assert mesh.topology.cells_of_faces.shape == (mesh.nfaces, 2)
    assert mesh.topology.faces_of_cells.min() >= 0
    assert mesh.topology.faces_of_cells.max() < mesh.nfaces

    return faces, faces_of_cells, cells_of_faces
