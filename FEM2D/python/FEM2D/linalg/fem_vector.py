import numpy as np


class FemVector:

    def __add__(self, other):
        if not isinstance(other, FemVector):
            return NotImplemented

        if self.names != other.names:
            raise ValueError(f"{self.names=} != {other.names=}")

        parts = tuple(
            U + V for U, V in zip(self.parts, other.parts)
        )

        return FemVector(
            parts,
            names=self.names,
            stack_storage=self.stack_storage,
        )

    def __sub__(self, other):
        if not isinstance(other, FemVector):
            return NotImplemented

        if self.names != other.names:
            raise ValueError(f"{self.names=} != {other.names=}")

        parts = tuple(
            U - V for U, V in zip(self.parts, other.parts)
        )

        return FemVector(
            parts,
            names=self.names,
            stack_storage=self.stack_storage,
        )

    def __mul__(self, alpha):
        if not np.isscalar(alpha):
            return NotImplemented

        parts = tuple(alpha * U for U in self.parts)

        return FemVector(
            parts,
            names=self.names,
            stack_storage=self.stack_storage,
        )

    def __rmul__(self, alpha):
        return self.__mul__(alpha)

    def __repr__(self):
        return f"FemVector(shape={self.shape}, names={self.names})"

    def __init__(self, parts, names=None, stack_storage=True):
        self.parts = tuple(np.asarray(U) for U in parts)
        self.stack_storage = stack_storage

        for U in self.parts:
            if U.ndim != 2:
                raise ValueError(f"FemVector parts must be 2D, got {U.shape=}")

        self.nparts = len(self.parts)

        if names is None:
            names = [f"u^{i}" for i in range(self.nparts)]
        self.names = tuple(names)
        self.name_to_part = {name: i for i, name in enumerate(self.names)}

    def __len__(self):
        return self.nparts

    def __getitem__(self, ipart):
        return self.part(ipart)

    def as_flat(self):
        return self.flatten()

    @classmethod
    def zeros(cls, ncomps, ndofs, names=None, stack_storage=True):
        ncomps = tuple(int(nc) for nc in ncomps)
        ndofs = tuple(int(n) for n in ndofs)

        if len(ncomps) != len(ndofs):
            raise ValueError(f"{len(ncomps)=} != {len(ndofs)=}")

        parts = [
            np.zeros((nc, n))
            for nc, n in zip(ncomps, ndofs)
        ]
        return cls(parts, names=names, stack_storage=stack_storage)

    @classmethod
    def from_flat(cls, u, ns, ncomps, names=None, stack_storage=True):
        ns = tuple(int(n) for n in ns)
        ncomps = tuple(int(nc) for nc in ncomps)

        if len(ns) != len(ncomps):
            raise ValueError(f"{len(ns)=} != {len(ncomps)=}")

        u = np.asarray(u)
        parts = []
        start = 0

        for n, nc in zip(ns, ncomps):
            n = int(n)
            nc = int(nc)
            size = n * nc
            chunk = u[start:start + size]

            if chunk.size != size:
                raise ValueError(f"Flat vector too short: need {size}, got {chunk.size}")

            if stack_storage:
                U = chunk.reshape(nc, n)
            else:
                U = chunk.reshape(n, nc).T

            parts.append(U)
            start += size

        if start != u.size:
            raise ValueError(f"Unused entries in flat vector: used {start}, total {u.size}")

        return cls(parts, names=names, stack_storage=stack_storage)

    def part_index(self, ipart):
        if isinstance(ipart, str):
            return self.name_to_part[ipart]
        return int(ipart)

    def part(self, ipart):
        return self.parts[self.part_index(ipart)]

    def copy(self):
        return FemVector(
            [U.copy() for U in self.parts],
            names=self.names,
            stack_storage=self.stack_storage,
        )

    def zeros_like(self):
        return FemVector(
            [np.zeros_like(U) for U in self.parts],
            names=self.names,
            stack_storage=self.stack_storage,
        )

    @property
    def size(self):
        return sum(U.size for U in self.parts)

    @property
    def ncomps(self):
        return np.array([U.shape[0] for U in self.parts], dtype=int)

    @property
    def ns(self):
        return np.array([U.shape[1] for U in self.parts], dtype=int)

    @property
    def shape(self):
        return tuple(U.shape for U in self.parts)

    def flatten(self):
        chunks = []
        for U in self.parts:
            if self.stack_storage:
                chunks.append(U.reshape(-1))
            else:
                chunks.append(U.T.reshape(-1))
        return np.concatenate(chunks)

    def ravel(self):
        return self.flatten()

    def __array__(self, dtype=None):
        u = self.flatten()
        if dtype is not None:
            u = u.astype(dtype, copy=False)
        return u

    def from_flat_like(self, x):
        return type(self).from_flat(
            x,
            ns=self.ns,
            ncomps=self.ncomps,
            names=self.names,
            stack_storage=self.stack_storage,
        )