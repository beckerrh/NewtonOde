import numpy as np


class VectorView:
    def __init__(self, ns, ncomps, names=None, stack_storage=True):
        self.ns = np.asarray(ns, dtype=int)
        self.ncomps = np.asarray(ncomps, dtype=int)

        if self.ns.shape != self.ncomps.shape:
            raise ValueError(f"{self.ns.shape=} != {self.ncomps.shape=}")

        self.nparts = len(self.ns)

        if names is None:
            names = [f"part{i}" for i in range(self.nparts)]
        if len(names) != self.nparts:
            raise ValueError(f"{len(names)=} != {self.nparts=}")

        self.names = tuple(names)
        self.name_to_part = {name: i for i, name in enumerate(self.names)}

        self.starts = np.zeros(self.nparts + 1, dtype=int)
        self.starts[1:] = np.cumsum(self.ns * self.ncomps)

        self.stack_storage = bool(stack_storage)

    def n(self):
        return int(self.starts[-1])

    def zeros(self, dtype=float):
        return np.zeros(self.n(), dtype=dtype)

    def zeros_like(self, u):
        return np.zeros_like(u)

    def copy(self, u):
        return np.copy(u)

    def part_index(self, ipart):
        if isinstance(ipart, str):
            return self.name_to_part[ipart]
        return int(ipart)

    def get_part(self, ipart, u):
        assert None
        ipart = self.part_index(ipart)
        return u[self.starts[ipart]:self.starts[ipart + 1]]

    def part(self, name, u):
        return self.get_part(name, u)

    def get_parts(self, u):
        return [self.get_part(i, u) for i in range(self.nparts)]

    def split(self, u):
        return tuple(np.split(u, self.starts[1:-1]))

    def get_norms(self, u):
        return [np.linalg.norm(self.get_part(i, u)) for i in range(self.nparts)]

    def get(self, ipart, icomp, u):
        ipart = self.part_index(ipart)
        base = self.get_part(ipart, u)
        ncomp = self.ncomps[ipart]

        if self.stack_storage:
            return base.reshape(ncomp, -1)[icomp]

        return base[icomp::ncomp]

    def set(self, ipart, icomp, u, v):
        self.get(ipart, icomp, u)[:] = v

    def add(self, ipart, icomp, u, s, v):
        self.get(ipart, icomp, u)[:] += s * v

    def col_indices(self, ipart, loc):
        """
        Return global column indices for a local connectivity array.

        loc has shape (ncells, nloc), e.g. facesOfCells for CR1.
        Output has shape (ncells*nloc, ncomp).
        """
        ipart = self.part_index(ipart)

        loc = np.asarray(loc, dtype=int)
        n = self.ns[ipart]
        ncomp = self.ncomps[ipart]
        start = self.starts[ipart]

        if self.stack_storage:
            return start + np.repeat(loc, ncomp).reshape(-1, ncomp) + n * np.arange(ncomp)

        return start + ncomp * np.repeat(loc, ncomp).reshape(-1, ncomp) + np.arange(ncomp)

    def scale(self, b, scales):
        for i in range(self.nparts):
            self.get_part(i, b)[:] = scales[i] @ self.get_part(i, b)

    def get_block(self, ipart, u):
        ipart = self.part_index(ipart)
        base = self.get_part(ipart, u)
        n = self.ns[ipart]
        ncomp = self.ncomps[ipart]

        if self.stack_storage:
            return base.reshape(ncomp, n)

        return base.reshape(n, ncomp).T

    def set_block(self, ipart, u, U):
        ipart = self.part_index(ipart)
        base = self.get_part(ipart, u)
        n = self.ns[ipart]
        ncomp = self.ncomps[ipart]

        U = np.asarray(U)
        if U.shape != (ncomp, n):
            raise ValueError(f"{U.shape=} != {(ncomp, n)=}")

        if self.stack_storage:
            base[:] = U.reshape(-1)
        else:
            base[:] = U.T.reshape(-1)