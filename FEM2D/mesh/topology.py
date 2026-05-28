import numpy as np


def construct_inner_faces(mesh):
    mesh.innerfaces = mesh.cells_of_faces[:, 1] >= 0
    mesh.cellsOfInteriorFaces = mesh.cells_of_faces[mesh.innerfaces]


def construct_faces_from_cells_vec(mesh, build_edge2face=True):
    """
    Vectorized construction of triangular mesh topology.

    Sets
    ----
    mesh.faces          : (nfaces, 2)
    mesh.faces_of_cells : (ncells, 3)
    mesh.cells_of_faces : (nfaces, 2)
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

    # Owner cell and local face number for each edge occurrence.
    owners = np.repeat(np.arange(ncells, dtype=np.int64), 3)
    local = np.tile(np.arange(3, dtype=np.int64), ncells)

    # ------------------------------------------------------------
    # Sort edges lexicographically.
    # Equal edges are now consecutive.
    # ------------------------------------------------------------
    order = np.lexsort((edges[:, 1], edges[:, 0]))

    edges_s = edges[order]
    owners_s = owners[order]
    local_s = local[order]

    # ------------------------------------------------------------
    # Detect first occurrence of every distinct edge.
    # ------------------------------------------------------------
    is_new = np.empty(edges_s.shape[0], dtype=bool)
    is_new[0] = True
    is_new[1:] = (
        (edges_s[1:, 0] != edges_s[:-1, 0])
        | (edges_s[1:, 1] != edges_s[:-1, 1])
    )

    face_ids_s = np.cumsum(is_new, dtype=np.int64) - 1
    nfaces = int(face_ids_s[-1]) + 1

    faces = edges_s[is_new].copy()

    # ------------------------------------------------------------
    # faces_of_cells[icell, iloc] = global face index
    # ------------------------------------------------------------
    faces_of_cells = np.empty((ncells, 3), dtype=np.int64)
    faces_of_cells[owners_s, local_s] = face_ids_s

    # ------------------------------------------------------------
    # cells_of_faces[iface] = [left_cell, right_cell or -1]
    # ------------------------------------------------------------
    first_pos = np.flatnonzero(is_new)
    counts = np.diff(np.r_[first_pos, edges_s.shape[0]])

    if np.any(counts > 2):
        bad = int(np.flatnonzero(counts > 2)[0])
        raise ValueError(
            f"non-manifold edge {tuple(faces[bad])}: "
            f"used by {int(counts[bad])} cells"
        )

    cells_of_faces = np.full((nfaces, 2), -1, dtype=np.int64)
    cells_of_faces[:, 0] = owners_s[first_pos]

    has_second = counts == 2
    second_pos = first_pos[has_second] + 1
    cells_of_faces[has_second, 1] = owners_s[second_pos]

    # ------------------------------------------------------------
    # Install on mesh.
    # ------------------------------------------------------------
    mesh.faces = faces
    mesh.faces_of_cells = faces_of_cells
    mesh.cells_of_faces = cells_of_faces
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
    assert mesh.faces.shape == (mesh.nfaces, 2)
    assert mesh.faces_of_cells.shape == (ncells, 3)
    assert mesh.cells_of_faces.shape == (mesh.nfaces, 2)
    assert mesh.faces_of_cells.min() >= 0
    assert mesh.faces_of_cells.max() < mesh.nfaces

    return faces, faces_of_cells, cells_of_faces
def construct_faces_from_cells_dict(mesh):
    cells = mesh.cells
    ncells = cells.shape[0]

    faces = []
    faces_of_cells = np.empty((ncells, 3), dtype=int)
    cells_of_faces = []
    edge2face = {}

    local_edges = ((1, 2), (2, 0), (0, 1))

    for icell, cell in enumerate(cells):
        for iloc, (a, b) in enumerate(local_edges):
            i = int(cell[a])
            j = int(cell[b])
            if i > j:
                i, j = j, i

            edge = (i, j)
            iface = edge2face.get(edge)

            if iface is None:
                iface = len(faces)
                edge2face[edge] = iface
                faces.append(edge)
                cells_of_faces.append([icell, -1])
            else:
                if cells_of_faces[iface][1] != -1:
                    raise ValueError(f"non-manifold edge {edge}")
                cells_of_faces[iface][1] = icell

            faces_of_cells[icell, iloc] = iface

    faces = np.asarray(faces, dtype=int)
    cells_of_faces = np.asarray(cells_of_faces, dtype=int)

    mesh.faces = faces
    mesh.faces_of_cells = faces_of_cells
    mesh.cells_of_faces = cells_of_faces
    mesh.edge2face = edge2face
    mesh.nfaces = faces.shape[0]

    return faces, faces_of_cells, cells_of_faces

def construct_faces_from_cells(mesh):
    cells = mesh.cells
    ncells = cells.shape[0]
    nnpc = cells.shape[1]

    nd = np.logical_not(np.eye(nnpc, dtype=bool)).ravel()

    if mesh.dimension == 2:
        allfaces = np.empty((ncells, 3, 2), dtype=cells.dtype)
        allfaces[:, 0, :] = np.sort(cells[:, [1, 2]], axis=1)  # opposite v0
        allfaces[:, 1, :] = np.sort(cells[:, [2, 0]], axis=1)  # opposite v1
        allfaces[:, 2, :] = np.sort(cells[:, [0, 1]], axis=1)  # opposite v2
        allfaces = allfaces.reshape(3 * ncells, 2)
    else:
        nd = np.logical_not(np.eye(nnpc, dtype=bool)).ravel()
        allfaces = np.sort(
            np.tile(cells, nnpc)[:, nd].reshape(ncells, nnpc, nnpc - 1),
            axis=2,
        ).reshape(nnpc * ncells, nnpc - 1)
        # allfaces = np.sort(
        #     np.tile(cells, nnpc)[:, nd].reshape(ncells, nnpc, nnpc - 1),
        #     axis=2,
        # ).reshape(nnpc * ncells, nnpc - 1)

    if mesh.dimension == 1:
        perm = np.argsort(allfaces, axis=0).ravel()
    else:
        dtype = ",".join([str(allfaces.dtype)] * (nnpc - 1))
        order = [f"f{i}" for i in range(nnpc - 1)]
        perm = np.argsort(allfaces.view(dtype), order=order, axis=0).ravel()

    allfaces_sorted = allfaces[perm]
    faces, indices = np.unique(allfaces_sorted, return_inverse=True, axis=0)

    faces_of_cells = np.zeros((ncells, nnpc), dtype=np.int64)

    locindex = np.tile(np.arange(nnpc), ncells).ravel()
    cellindex = np.repeat(np.arange(ncells), nnpc)

    faces_of_cells[cellindex[perm], locindex[perm]] = indices

    flat_faces = faces_of_cells.ravel()
    flat_cells = np.repeat(np.arange(ncells), nnpc)

    order2 = np.argsort(flat_faces)
    sf = flat_faces[order2]
    sc = flat_cells[order2]

    counts = np.bincount(sf, minlength=faces.shape[0])
    starts = np.r_[0, np.cumsum(counts[:-1])]

    first_cell = sc[starts]
    second_cell = -np.ones(faces.shape[0], dtype=np.int64)

    mask = counts == 2
    second_cell[mask] = sc[starts[mask] + 1]

    mesh.cells_of_faces = np.vstack([first_cell, second_cell]).T
    mesh.faces = faces
    mesh.nfaces = faces.shape[0]
    mesh.faces_of_cells = faces_of_cells
