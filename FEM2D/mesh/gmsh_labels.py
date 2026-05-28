import numpy as np


def attach_gmsh_labels(mesh, meshio_mesh, simplex_name, face_name):
    celltypes = [c.type for c in meshio_mesh.cells]

    cellsoflabel_by_type, names = parse_cell_sets(
        cell_sets=meshio_mesh.cell_sets,
        cells_dict=meshio_mesh.cells_dict,
        celltypes=celltypes,
    )

    mesh.labels.cell = cellsoflabel_by_type.get(simplex_name, {})
    mesh.labels.names.update(names)

    mesh.labels.vertex = cellsoflabel_by_type.get("vertex", {})
    mesh.labels.line = cellsoflabel_by_type.get("line", {})

    if face_name not in cellsoflabel_by_type:
        raise ValueError(f"{face_name=} not found in physical labels.")

    construct_boundary_labels(
        mesh,
        faces_gmsh=meshio_mesh.cells_dict[face_name],
        physlabels_gmsh=cellsoflabel_by_type[face_name],
    )

def parse_cell_sets(cell_sets, cells_dict, celltypes):
    sizes = {key: 0 for key in celltypes}
    cellsoflabel = {key: {} for key in celltypes}
    ctordered = []

    names = {}
    name_to_id = {}
    labind = 0

    for label, cb in cell_sets.items():
        if label == "gmsh:bounding_entities":
            continue

        if len(cb) != len(celltypes):
            raise KeyError(f"Mismatch in cell_sets: {label=}, {celltypes=}")

        for celltype, info in zip(celltypes, cb):
            if info is None:
                continue

            try:
                ilabel = int(label)
            except Exception:
                if label in name_to_id:
                    ilabel = name_to_id[label]
                else:
                    labind -= 1
                    ilabel = labind
                    name_to_id[label] = ilabel
                    names.setdefault(celltype, {})[ilabel] = label

            cellsoflabel[celltype][ilabel] = np.asarray(info, dtype=np.int64).copy()
            sizes[celltype] += info.shape[0]
            ctordered.append(celltype)

    # Correct numbering in gmsh cell_sets.
    n = 0
    for ct in list(dict.fromkeys(ctordered)):
        for _, cb in cellsoflabel[ct].items():
            cb -= n
        n += sizes[ct]

    return cellsoflabel, names

def construct_boundary_labels(mesh, faces_gmsh, physlabels_gmsh):
    faces_gmsh = np.sort(np.asarray(faces_gmsh, dtype=np.int64), axis=1)

    bdryids = np.flatnonzero(mesh.cells_of_faces[:, 1] == -1)
    bdryfaces = np.sort(mesh.faces[bdryids], axis=1)

    nnpc = mesh.topology.cells.shape[1]
    dtype_g = ", ".join([str(faces_gmsh.dtype)] * (nnpc - 1))
    dtype_b = ", ".join([str(bdryfaces.dtype)] * (nnpc - 1))
    order = [f"f{i}" for i in range(nnpc - 1)]

    if mesh.dimension == 1:
        bp = np.argsort(faces_gmsh.view(dtype_g), axis=0).ravel()
        fp = np.argsort(bdryfaces.view(dtype_b), axis=0).ravel()
    else:
        bp = np.argsort(faces_gmsh.view(dtype_g), order=order, axis=0).ravel()
        fp = np.argsort(bdryfaces.view(dtype_b), order=order, axis=0).ravel()

    bpi = np.argsort(bp)

    indices = (faces_gmsh[bp, None] == bdryfaces[fp]).all(axis=-1).any(axis=-1)

    bp2 = bp[indices]
    binv = np.empty_like(bp)
    binv[bp2] = np.arange(len(bp2))

    mesh.labels.boundary = {}

    for color, cb in physlabels_gmsh.items():
        cb = np.asarray(cb, dtype=np.int64)

        if len(cb) == 0:
            continue

        # Only keep physical faces which are actual boundary faces.
        if np.all(indices[bpi[cb]]):
            mesh.labels.boundary[int(color)] = bdryids[fp[binv[cb]]]
        else:
            # This happens for physical lines used only to help gmsh,
            # e.g. internal hole construction or interface labels.
            assert not indices[bpi[cb[0]]]