import numpy as np

#=================================================================#
class VectorView():
    def __init__(self, **kwargs):
        self.ncomps = np.asarray(kwargs.pop('ncomps'))
        self.nparts = len(self.ncomps)
        self.ns = np.asarray(kwargs.pop('ns'))
        self.starts = np.zeros(len(self.ns)+1, dtype=self.ns.dtype)
        self.starts[1:] = np.cumsum(self.ns*self.ncomps)
        self.stack_storage = kwargs.pop('stack_storage')
    def n(self): return self.starts[-1]
    def split(self, u): return np.split(u, self.starts[1:-1])
    def get_norms(self, u):
        return [np.linalg.norm(u[self.starts[i]: self.starts[i+1]]) for i in range(self.nparts)]
    def get_parts(self, u):
        return [u[self.starts[i]: self.starts[i+1]] for i in range(self.nparts)]
    def get_part(self, ipart, u):
        return u[self.starts[ipart]: self.starts[ipart+1]]
    def get(self, ipart, icomp, u):
        if self.stack_storage:
            return self.get_part(ipart, u).reshape(self.ncomps[ipart],-1)[icomp]
            # return u[self.starts[ipart] +icomp*self.ns[ipart]: self.starts[ipart] +(icomp+1)*self.ns[ipart]]
        return u[self.starts[ipart]+icomp: self.starts[ipart+1]:self.ncomps[ipart]]
    def set(self, ipart, icomp, u, v):
        if self.stack_storage:
            self.get_part(ipart, u).reshape(self.ncomps[ipart],-1)[icomp] = v
            return
        u[self.starts[ipart]+icomp: self.starts[ipart+1]:self.ncomps[ipart]] = v
    def add(self, ipart, icomp, u, s, v):
        if self.stack_storage:
            self.get_part(ipart, u).reshape(self.ncomps[ipart],-1)[icomp] += s*v
            return
        u[self.starts[ipart]+icomp: self.starts[ipart+1]:self.ncomps[ipart]] += s*v
    def col_indices(self, ipart, foc, ncells, nloc, nfaces):
        ncomp = self.ncomps[ipart]
        if self.stack_storage:
            return np.repeat(foc, ncomp).reshape(ncells * nloc, ncomp) + nfaces*np.arange(ncomp)
        else:
            return ncomp*np.repeat(foc, ncomp).reshape(ncells * nloc, ncomp) + np.arange(ncomp)
    def scale(self, b , scales):
        for i in range(self.nparts):
            b[self.starts[i]: self.starts[i+1]] = scales[i]@b[self.starts[i]: self.starts[i+1]]

