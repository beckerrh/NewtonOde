import numpy as np
import scipy.sparse
from . import barycentric
from ..linalg.fem_vector import FemVector


#=================================================================#
class MeshTransfer:
    def __init__(self, transfers, ncomps, names=None):
        self.transfers = tuple(transfers)
        self.ncomps = tuple(int(nc) for nc in ncomps)
        self.names = None if names is None else tuple(names)

        if len(self.transfers) != len(self.ncomps):
            raise ValueError(f"{len(self.transfers)=} != {len(self.ncomps)=}")

    def prolong(self, uc):
        if isinstance(uc, FemVector):
            parts = [T.prolong(Uc) for T, Uc in zip(self.transfers, uc.parts)]
            return FemVector(parts, names=uc.names, stack_storage=uc.stack_storage)

        if len(self.transfers) != 1:
            raise TypeError("Flat ndarray transfer only implemented for one block")

        nc = self.ncomps[0]
        T = self.transfers[0]

        ncoarse = T.P.shape[1]
        Uc = np.asarray(uc).reshape(nc, ncoarse)
        Uf = T.prolong(Uc)
        return Uf.reshape(-1)

    def restrict(self, uf):
        if isinstance(uf, FemVector):
            parts = [T.restrict(Uf) for T, Uf in zip(self.transfers, uf.parts)]
            return FemVector(parts, names=uf.names, stack_storage=uf.stack_storage)

        if len(self.transfers) != 1:
            raise TypeError("Flat ndarray transfer only implemented for one block")

        nc = self.ncomps[0]
        T = self.transfers[0]

        nfine = T.P.shape[0]
        Uf = np.asarray(uf).reshape(nc, nfine)
        Uc = T.restrict(Uf)
        return Uc.reshape(-1)

    def interpolate(self, uc):
        if isinstance(uc, FemVector):
            parts = [
                T.interpolate(Uc)
                for T, Uc in zip(self.transfers, uc.parts)
            ]
            return FemVector(parts, names=uc.names, stack_storage=uc.stack_storage)

        if len(self.transfers) != 1:
            raise TypeError("Flat ndarray interpolation only implemented for one block")

        nc = self.ncomps[0]
        T = self.transfers[0]

        # coarse dofs: prefer P if available
        if T.P is not None:
            ncoarse = T.P.shape[1]
        else:
            if uc.size % nc != 0:
                raise ValueError(f"Cannot reshape {uc.size=} with {nc=}")
            ncoarse = uc.size // nc

        Uc = np.asarray(uc).reshape(nc, ncoarse)
        Uf = T.interpolate(Uc)
        return Uf.reshape(-1)

#=================================================================#
class BlockTransfer:
    def __init__(self, P=None, R=None, interpolate=None):
        self.P = None if P is None else P.tocsr()
        self.R = None if R is None else R.tocsr()
        self._interpolate = interpolate

    def prolong(self, Uc):
        if self.P is None:
            return self.interpolate(Uc)
        return Uc @ self.P.T

    def restrict(self, Uf):
        if self.R is not None:
            return Uf @ self.R.T
        if self.P is None:
            raise ValueError("Restriction requires P or R")
        return Uf @ self.P

    def interpolate(self, Uc):
        if self._interpolate is not None:
            return self._interpolate(Uc)
        if self.P is None:
            raise ValueError("Interpolation requires interpolate or P")
        return self.prolong(Uc)

#=================================================================#
def p1_prolongation(info):
    rows, cols, data = [], [], []

    for i in range(info.old_npoints):
        rows.append(i)
        cols.append(i)
        data.append(1.0)

    for m, (a, b) in info.midpoint_parents.items():
        rows.extend([m, m])
        cols.extend([a, b])
        data.extend([0.5, 0.5])

    return scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(info.new_npoints, info.old_npoints),
    )
def cr1_interpolate(info, u_old):
    parent = info.parent_cell_of_face
    order = np.argsort(parent)
    parent_sorted = parent[order]

    u_new = np.empty(info.new_nfaces, dtype=u_old.dtype)

    start = 0
    while start < order.size:
        c_old = parent_sorted[start]
        end = start + 1
        while end < order.size and parent_sorted[end] == c_old:
            end += 1

        faces = order[start:end]

        x = info.new_face_centers[faces]
        simplex = info.old_points[info.old_cells[c_old]]

        lam = barycentric.coords(x, simplex)
        phi = 1.0 - 2.0 * lam

        old_faces = info.old_faces_of_cells[c_old]
        u_new[faces] = phi @ u_old[old_faces]

        start = end

    return u_new
def cr1_prolongation(info):
    parent = info.parent_cell_of_face
    order = np.argsort(parent)
    parent_sorted = parent[order]

    n = info.new_nfaces
    rows = np.empty(3 * n, dtype=int)
    cols = np.empty(3 * n, dtype=int)
    data = np.empty(3 * n, dtype=float)

    k = 0
    start = 0
    while start < order.size:
        c_old = parent_sorted[start]
        end = start + 1
        while end < order.size and parent_sorted[end] == c_old:
            end += 1

        faces = order[start:end]
        m = faces.size

        x = info.new_face_centers[faces]
        simplex = info.old_points[info.old_cells[c_old]]

        lam = barycentric.coords(x, simplex)
        phi = 1.0 - 2.0 * lam

        old_faces = info.old_faces_of_cells[c_old]

        rows[k:k+3*m] = np.repeat(faces, 3)
        cols[k:k+3*m] = np.tile(old_faces, m)
        data[k:k+3*m] = phi.ravel()

        k += 3*m
        start = end

    return scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(info.new_nfaces, info.old_nfaces),
    )