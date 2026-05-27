# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import numpy as np
import numpy.linalg as linalg
import scipy.sparse as sparse
from FEM2D.mesh.simplex_mesh import SimplexMesh


#=================================================================#
# class P1general(fems.Fem):
class P1general():
    def __init__(self, **kwargs):
        pass
    def __repr__(self):
        s = self.__class__.__name__
        if hasattr(self, 'mesh'): s+= " (" +str(self.mesh) + ")"
        return s
    def setMesh(self, mesh, innersides=False):
        self.mesh = mesh
        self.nloc = self.nlocal()
        if innersides: self.mesh.constructInnerFaces()
    def computeStencilCell(self, dofspercell):
        self.cols = np.tile(dofspercell, self.nloc).ravel()
        self.rows = np.repeat(dofspercell, self.nloc).ravel()
    def computeMatrixDiffusion(self, coeff, coeffM=None):
        ndofs = self.nunknowns()
        cellgrads = self.cellgrads[:,:,:self.mesh.dimension]
        mat = np.einsum('n,nil,njl->nij', self.mesh.cell_volumes*coeff, cellgrads, cellgrads)
        if coeffM: mat += self._computeMassMatrix(coeff=coeffM)
        return sparse.coo_matrix((mat.ravel(), (self.rows, self.cols)), shape=(ndofs, ndofs)).tocsr()
    def computeFormDiffusion(self, du, u, coeff):
        doc = self.dofspercell()
        cellgrads = self.cellgrads[:,:,:self.mesh.dimension]
        r = np.einsum('n,nil,njl,nj->ni', self.mesh.cell_volumes*coeff, cellgrads, cellgrads, u[doc])
        np.add.at(du, doc, r)
    def computeMatrixLps(self, betart, lpsparam=0.1):
        dimension, dV, ndofs, nloc, dofspercell = self.mesh.dimension, self.mesh.cell_volumes, self.nunknowns(), self.nlocal(), self.dofspercell()
        if not hasattr(self.mesh,'innerfaces'): self.mesh.constructInnerFaces()
        ci = self.mesh.cellsOfInteriorFaces
        ci0, ci1 = ci[:,0], ci[:,1]
        normalsS = self.mesh.normals[self.mesh.innerfaces]
        dS = linalg.norm(normalsS, axis=1)
        scale = 0.5*(dV[ci0]+ dV[ci1])
        betan = np.absolute(betart[self.mesh.innerfaces])
        # betan = 0.5*(np.linalg.norm(betaC[ci0],axis=1)+ np.linalg.norm(betaC[ci1],axis=1))
        scale *= lpsparam*dS*betan
        cg0 = self.cellgrads[ci0, :, :]
        cg1 = self.cellgrads[ci1, :, :]
        mat00 = np.einsum('nki,nli,n->nkl', cg0, cg0, scale)
        mat01 = np.einsum('nki,nli,n->nkl', cg0, cg1, -scale)
        mat10 = np.einsum('nki,nli,n->nkl', cg1, cg0, -scale)
        mat11 = np.einsum('nki,nli,n->nkl', cg1, cg1, scale)
        rows0 = dofspercell[ci0,:].repeat(nloc)
        cols0 = np.tile(dofspercell[ci0,:],nloc).reshape(-1)
        rows1 = dofspercell[ci1,:].repeat(nloc)
        cols1 = np.tile(dofspercell[ci1,:],nloc).reshape(-1)
        A00 = sparse.coo_matrix((mat00.reshape(-1), (rows0, cols0)), shape=(ndofs, ndofs))
        A01 = sparse.coo_matrix((mat01.reshape(-1), (rows0, cols1)), shape=(ndofs, ndofs))
        A10 = sparse.coo_matrix((mat10.reshape(-1), (rows1, cols0)), shape=(ndofs, ndofs))
        A11 = sparse.coo_matrix((mat11.reshape(-1), (rows1, cols1)), shape=(ndofs, ndofs))
        return A00+A01+A10+A11
    def computeFormLps(self, du, u, betart, lpsparam=0.1):
        # assert 0
        dimension, dV, ndofs, nloc, dofspercell = self.mesh.dimension, self.mesh.cell_volumes, self.nunknowns(), self.nlocal(), self.dofspercell()
        ci = self.mesh.cellsOfInteriorFaces
        ci0, ci1 = ci[:,0], ci[:,1]
        normalsS = self.mesh.normals[self.mesh.innerfaces]
        dS = linalg.norm(normalsS, axis=1)
        scale = 0.5*(dV[ci0]+ dV[ci1])
        betan = np.absolute(betart[self.mesh.innerfaces])
        scale *= lpsparam*dS*betan
        cg0 = self.cellgrads[ci0, :, :]
        cg1 = self.cellgrads[ci1, :, :]
        mat = np.einsum('nki,nli,n,nl->nk', cg0, cg0, +scale, u[dofspercell[ci0,:]])
        np.add.at(du, dofspercell[ci0,:], mat)
        mat = np.einsum('nki,nli,n,nl->nk', cg0, cg1, -scale, u[dofspercell[ci1,:]])
        np.add.at(du, dofspercell[ci0,:], mat)
        mat = np.einsum('nki,nli,n,nl->nk', cg1, cg0, -scale, u[dofspercell[ci0,:]])
        np.add.at(du, dofspercell[ci1,:], mat)
        mat = np.einsum('nki,nli,n,nl->nk', cg1, cg1, +scale, u[dofspercell[ci1,:]])
        np.add.at(du, dofspercell[ci1,:], mat)
    def computeFormTransportCellWise(self, du, u, data, type):
        beta, betart = data.betacell, data.betart
        ndofs, dim, dV, dofspercell = self.nunknowns(), self.mesh.dimension, self.mesh.cell_volumes, self.dofspercell()
        cellgrads = self.cellgrads[:,:,:dim]
        if type=='centered':
            mat = np.einsum('n,njk,nk,i,nj -> ni', dV, cellgrads, beta, 1/(dim+1)*np.ones(dim+1),u[dofspercell])
        elif type=='supg':
            mus = data.md.mus
            mat = np.einsum('n,njk,nk,ni,nj -> ni', dV, cellgrads, beta, 1-dim*mus,u[dofspercell])
        else: raise ValueError(f"unknown type {type=}")
        np.add.at(du, dofspercell, mat)
        self.massDotBoundary(du, u, coeff=-np.minimum(betart, 0))
    def computeMatrixTransportCellWise(self, data, type):
        beta, betart = data.betacell, data.betart
        ndofs, dim, dV, dofspercell = self.nunknowns(), self.mesh.dimension, self.mesh.cell_volumes, self.dofspercell()
        # nfaces, dim, dV = self.mesh.nfaces, self.mesh.dimension, self.mesh.cell_volumes
        cellgrads = self.cellgrads[:,:,:dim]
        if type=='centered':
            # betagrad = np.einsum('njk,nk -> nj', cellgrads, beta)
            mat = np.einsum('n,njk,nk,i -> nij', dV, cellgrads, beta, 1/(dim+1)*np.ones(dim+1))
            # mat += np.einsum('n,nj,ni -> nij', dV*deltas, betagrad, betagrad)
        elif type=='supg':
            mus = data.md.mus
            mat = np.einsum('n,njk,nk,ni -> nij', dV, cellgrads, beta, 1-dim*mus)
        else: raise ValueError(f"unknown type {type=}")
        A = sparse.coo_matrix((mat.ravel(), (self.rows, self.cols)), shape=(ndofs, ndofs))
        return A - self.computeBdryMassMatrix(coeff=np.minimum(betart, 0))

# ====================================================================================

#------------------------------
def test(self):
    import scipy.sparse.linalg as splinalg
    colors = self.mesh.labels.boundary.keys()
    bdrydata = self.prepareBoundary(colorsdir=colors)
    A = self.computeMatrixDiffusion(coeff=1)
    A = self.matrixBoundaryStrong(A, bdrydata=bdrydata)
    b = np.zeros(self.nunknowns())
    rhs = np.vectorize(lambda x,y,z: 1)
    b = self.computeRhsCell(b, rhs)
    self.vectorBoundaryStrongZero(b, bdrydata)
    return self.tonode(splinalg.spsolve(A, b))

# ------------------------------------- #

if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
