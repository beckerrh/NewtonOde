import numpy as np


def construct_inner_faces(mesh):
    mesh.innerfaces = mesh.cells_of_faces[:, 1] >= 0
    mesh.cellsOfInteriorFaces = mesh.cells_of_faces[mesh.innerfaces]


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

    unique, indices_unique = np.unique(faces_of_cells, return_index=True)
    assert np.all(unique == np.arange(faces.shape[0]))

    i0, i1 = np.unravel_index(indices_unique, shape=faces_of_cells.shape)

    foc = faces_of_cells.copy()
    foc[i0, i1] = -1

    unique2, indices2 = np.unique(foc, return_index=True)
    i2, _ = np.unravel_index(indices2[1:], shape=foc.shape)

    second_cell = -np.ones(faces.shape[0], dtype=np.int64)
    second_cell[unique2[1:]] = i2

    mesh.faces = faces
    mesh.nfaces = faces.shape[0]
    mesh.faces_of_cells = faces_of_cells
    mesh.cells_of_faces = np.vstack([i0, second_cell]).T

    # old aliases
    mesh.facesOfCells = mesh.faces_of_cells
    mesh.cellsOfFaces = mesh.cells_of_faces

    if mesh.dimension == 2:
        for ic, (v0, v1, v2) in enumerate(cells):
            expected = [
                tuple(sorted((v1, v2))),
                tuple(sorted((v2, v0))),
                tuple(sorted((v0, v1))),
            ]
            for iloc in range(3):
                f = faces_of_cells[ic, iloc]
                got = tuple(mesh.faces[f])
                if got != expected[iloc]:
                    raise RuntimeError(
                        f"bad local face order: cell={ic}, iloc={iloc}, "
                        f"got={got}, expected={expected[iloc]}"
                    )