import numpy as np


def construct_centers(mesh):
    mesh.cell_centers = mesh.points[mesh.cells].mean(axis=1)
    mesh.face_centers = mesh.points[mesh.faces].mean(axis=1)


def construct_normals_and_volumes(mesh):
    elem = mesh.cells
    points = mesh.points

    if mesh.dimension == 1:
        x = points[:, 0]
        mesh.normals = np.stack(
            (
                np.ones(mesh.faces.shape[0]),
                np.zeros(mesh.faces.shape[0]),
                np.zeros(mesh.faces.shape[0]),
            ),
            axis=-1,
        )
        dx1 = x[elem[:, 1]] - x[elem[:, 0]]
        mesh.cell_volumes = np.abs(dx1)

    elif mesh.dimension == 2:
        x, y = points[:, 0], points[:, 1]

        sidesx = x[mesh.faces[:, 1]] - x[mesh.faces[:, 0]]
        sidesy = y[mesh.faces[:, 1]] - y[mesh.faces[:, 0]]
        mesh.normals = np.stack((-sidesy, sidesx, np.zeros(mesh.faces.shape[0])), axis=-1)

        dx1 = x[elem[:, 1]] - x[elem[:, 0]]
        dx2 = x[elem[:, 2]] - x[elem[:, 0]]
        dy1 = y[elem[:, 1]] - y[elem[:, 0]]
        dy2 = y[elem[:, 2]] - y[elem[:, 0]]

        mesh.cell_volumes = 0.5 * np.abs(dx1 * dy2 - dx2 * dy1)

    elif mesh.dimension == 3:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        x1 = x[mesh.faces[:, 1]] - x[mesh.faces[:, 0]]
        y1 = y[mesh.faces[:, 1]] - y[mesh.faces[:, 0]]
        z1 = z[mesh.faces[:, 1]] - z[mesh.faces[:, 0]]

        x2 = x[mesh.faces[:, 2]] - x[mesh.faces[:, 0]]
        y2 = y[mesh.faces[:, 2]] - y[mesh.faces[:, 0]]
        z2 = z[mesh.faces[:, 2]] - z[mesh.faces[:, 0]]

        sidesx = y1 * z2 - y2 * z1
        sidesy = x2 * z1 - x1 * z2
        sidesz = x1 * y2 - x2 * y1

        mesh.normals = 0.5 * np.stack((sidesx, sidesy, sidesz), axis=-1)

        dx1 = x[elem[:, 1]] - x[elem[:, 0]]
        dx2 = x[elem[:, 2]] - x[elem[:, 0]]
        dx3 = x[elem[:, 3]] - x[elem[:, 0]]

        dy1 = y[elem[:, 1]] - y[elem[:, 0]]
        dy2 = y[elem[:, 2]] - y[elem[:, 0]]
        dy3 = y[elem[:, 3]] - y[elem[:, 0]]

        dz1 = z[elem[:, 1]] - z[elem[:, 0]]
        dz2 = z[elem[:, 2]] - z[elem[:, 0]]
        dz3 = z[elem[:, 3]] - z[elem[:, 0]]

        mesh.cell_volumes = (1.0 / 6.0) * np.abs(
            dx1 * (dy2 * dz3 - dy3 * dz2)
            - dx2 * (dy1 * dz3 - dy3 * dz1)
            + dx3 * (dy1 * dz2 - dy2 * dz1)
        )

    else:
        raise ValueError(f"Unsupported dimension {mesh.dimension}")

    orient_normals(mesh)


def orient_normals(mesh):
    ind = np.arange(mesh.cells.shape[0])

    mesh.sigma = (
        2
        * np.equal(
            mesh.cells_of_faces[mesh.faces_of_cells[ind, :], 0],
            ind[:, None],
        )
        - 1
    )

    # Boundary normals: exterior orientation.
    ib = np.arange(mesh.faces.shape[0])[mesh.cells_of_faces[:, 1] == -1]
    xt = (
        np.mean(mesh.points[mesh.faces[ib]], axis=1)
        - np.mean(mesh.points[mesh.cells[mesh.cells_of_faces[ib, 0]]], axis=1)
    )
    m = np.einsum("nk,nk->n", mesh.normals[ib], xt) < 0
    mesh.normals[ib[m]] *= -1

    # Interior normals: from cell 0 to cell 1.
    ii = np.arange(mesh.faces.shape[0])[mesh.cells_of_faces[:, 1] != -1]
    xt = (
        np.mean(mesh.points[mesh.cells[mesh.cells_of_faces[ii, 1]]], axis=1)
        - np.mean(mesh.points[mesh.cells[mesh.cells_of_faces[ii, 0]]], axis=1)
    )
    m = np.einsum("nk,nk->n", mesh.normals[ii], xt) < 0
    mesh.normals[ii[m]] *= -1