# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import numpy as np
import numpy.linalg as linalg
import scipy.sparse as sparse
from ..mesh import SimplexMesh
from . import barycentric

#=================================================================#
class P1general():
    def __init__(self, **kwargs):
        pass
    def __repr__(self):
        s = self.__class__.__name__
        if hasattr(self, 'mesh'): s+= " (" +str(self.mesh) + ")"
        return s

    def check_form_matrix(self, A, form, u):
        Fu = np.zeros_like(u)
        form(Fu, u)
        return np.linalg.norm(Fu - A @ u)

    def _xyz_from_points(self, points):
        x = points[:, 0]
        y = points[:, 1] if points.shape[1] > 1 else np.zeros_like(x)
        z = points[:, 2] if points.shape[1] > 2 else np.zeros_like(x)
        return x, y, z

    # P1general.setMesh
    def setMesh(self, mesh, innersides=False):
        self.mesh = mesh
        self.dim = mesh.dimension
        self.nloc = self.nlocal()

        if innersides:
            self.mesh.construct_inner_faces()

        self.cellgrads = self.computeCellGrads()

        self.cell_x, self.cell_y, self.cell_z = self._xyz_from_points(
            mesh.geometry.cell_centers
        )
    def cell_grad(self, u):
        dofs = self.dofs_of_cells()
        dim = self.mesh.dimension
        return np.einsum(
            "nij,ni->nj",
            self.cellgrads[:, :, :dim],
            u[dofs],
        )
    def computeStencilCell(self, dofspercell):
        self.cols = np.tile(dofspercell, self.nloc).ravel()
        self.rows = np.repeat(dofspercell, self.nloc).ravel()

    def boundary_faces_from_bdrydata(self, bdrydata=None):
        if bdrydata is None:
            bdrydata = self.bdrydata

        if bdrydata is None:
            return np.array([], dtype=int)

        labels = getattr(bdrydata, "labels", None)

        if labels is None:
            faces = []
            for arr in self.mesh.labels.boundary.values():
                faces.extend(arr)
            return np.unique(faces).astype(int)

        faces = []
        for lab in labels:
            faces.extend(self.mesh.labels.boundary[lab])

        return np.unique(faces).astype(int)


    def computeErrorL2(self, solexact, uh):
        x, y, z = self.dof_coordinates_xyz()

        en = solexact(x, y, z) - uh
        Men = np.zeros_like(en)

        return np.sqrt(np.dot(en, self.massDot(Men, en))), en

    def computeErrorL2Cell(self, solexact, uh):
        xc, yc, zc = self.cell_x, self.cell_y, self.cell_z

        uhc = np.mean(
            uh[self.dofs_of_cells()],
            axis=1,
        )

        e = uhc - solexact(xc, yc, zc)
        ec = self.mesh.geometry.cell_volumes * e ** 2

        return np.sqrt(ec.sum()), ec

    def computeErrorFluxL2(self, solexact, uh, diffcell=None):
        xc, yc, zc = self.cell_x, self.cell_y, self.cell_z
        graduh = self.cell_grad(uh)

        err2 = 0.0

        for d in range(self.mesh.dimension):
            e = solexact.d(d, xc, yc, zc) - graduh[:, d]

            if diffcell is None:
                err2 += np.sum(e ** 2 * self.mesh.geometry.cell_volumes)
            else:
                err2 += np.sum(diffcell * e ** 2 * self.mesh.geometry.cell_volumes)

        return np.sqrt(err2)

    def computeMatrixDiffusion(self, coeff):
        ndofs = self.nunknowns()
        cellgrads = self.cellgrads[:,:,:self.mesh.dimension]
        mat = np.einsum('n,nil,njl->nij', self.mesh.geometry.cell_volumes*coeff, cellgrads, cellgrads)
        return sparse.coo_matrix((mat.ravel(), (self.rows, self.cols)), shape=(ndofs, ndofs)).tocsr()
    def computeFormDiffusion(self, du, u, coeff):
        doc = self.dofspercell()
        cellgrads = self.cellgrads[:,:,:self.mesh.dimension]
        r = np.einsum('n,nil,njl,nj->ni', self.mesh.geometry.cell_volumes*coeff, cellgrads, cellgrads, u[doc])
        np.add.at(du, doc, r)

    def computeMassMatrix(self, coeff=1, lumped=False):
        if lumped:
            return self.computeMassMatrixLumped(coeff)
        return self.computeMassMatrixExact(coeff)

    def computeMassMatrixExact(self, coeff=1):
        dim = self.mesh.dimension
        dV = self.mesh.geometry.cell_volumes
        ndofs = self.nunknowns()
        cells = self.mesh.topology.cells
        coeff = np.asarray(coeff)

        if coeff.ndim == 0:
            massloc = np.einsum("n,ij->nij", coeff * dV, self.masslocal())

        elif coeff.shape == (self.mesh.ncells,):
            massloc = np.einsum("n,ij->nij", coeff * dV, self.masslocal())

        elif coeff.shape == (ndofs,):
            c = coeff[self.dofs_of_cells()]
            T3 = self.masslocal_variable()
            massloc = dV[:, None, None] * np.einsum("nm,mij->nij", c, T3)

        else:
            raise ValueError(
                "exact mass expects scalar, cellwise, or dofwise coefficient, "
                f"got {coeff.shape=}, {self.mesh.ncells=}, {ndofs=}"
            )

        return sparse.coo_matrix(
            (massloc.ravel(), (self.rows, self.cols)),
            shape=(ndofs, ndofs),
        ).tocsr()

    def computeMassMatrixLumped(self, coeff=1):
        dim = self.mesh.dimension
        dV = self.mesh.geometry.cell_volumes
        ndofs = self.nunknowns()
        dofs = self.dofs_of_cells()
        coeff = np.asarray(coeff)

        rows = dofs.ravel()

        if coeff.ndim == 0:
            mass = coeff * dV.repeat(dim + 1) / (dim + 1)

        elif coeff.shape == (self.mesh.ncells,):
            mass = (coeff * dV / (dim + 1)).repeat(dim + 1)

        else:
            raise ValueError(
                "lumped reaction mass expects scalar or cellwise coefficient, "
                f"got {coeff.shape=}, {self.mesh.ncells=}"
            )

        return sparse.coo_matrix(
            (mass, (rows, rows)),
            shape=(ndofs, ndofs),
        ).tocsr()

    def computeMassMatrixCellReaction(self, coeff=1, weights=None):
        dim = self.mesh.dimension
        nloc = dim + 1
        dV = self.mesh.geometry.cell_volumes
        ndofs = self.nunknowns()
        dofs = self.dofs_of_cells()
        coeff = np.asarray(coeff)

        if coeff.ndim == 0:
            coeff = coeff * np.ones(self.mesh.ncells)

        if coeff.shape != (self.mesh.ncells,):
            raise ValueError(
                f"expected scalar or cellwise coeff, got {coeff.shape=}, "
                f"{self.mesh.ncells=}"
            )

        if weights is None:
            weights = np.full(nloc, 1.0 / nloc)

        weights = np.asarray(weights)

        weights = np.asarray(weights)

        local = np.empty(
            (self.mesh.ncells, nloc, nloc),
            dtype=np.result_type(coeff, dV),
        )

        local[:] = (
                dV[:, None, None]
                * coeff[:, None, None]
                * weights[None, None, :]
                / nloc
        )

        rows = np.repeat(dofs, nloc, axis=1).ravel()
        cols = np.tile(dofs, (1, nloc)).ravel()

        assert local.size == rows.size == cols.size

        return sparse.coo_matrix(
            (local.ravel(), (rows, cols)),
            shape=(ndofs, ndofs),
        ).tocsr()

    def computeMassMatrixCellAverageReaction(self, coeff=1):
        dim = self.mesh.dimension
        dV = self.mesh.geometry.cell_volumes
        ndofs = self.nunknowns()
        dofs = self.dofs_of_cells()
        coeff = np.asarray(coeff)

        if coeff.ndim == 0:
            coeff = coeff * np.ones(self.mesh.ncells)

        if coeff.shape != (self.mesh.ncells,):
            raise ValueError(f"expected cellwise coeff, got {coeff.shape=}")

        nloc = dim + 1
        local = (coeff * dV / (nloc * nloc))[:, None, None] * np.ones(
            (self.mesh.ncells, nloc, nloc)
        )

        rows = np.repeat(dofs, nloc, axis=1).ravel()
        cols = np.tile(dofs, (1, nloc)).ravel()

        return sparse.coo_matrix(
            (local.ravel(), (rows, cols)),
            shape=(ndofs, ndofs),
        ).tocsr()

    # def computeMassMatrixLumped(self, coeff=1):
    #     dim = self.mesh.dimension
    #     dV = self.mesh.geometry.cell_volumes
    #     ndofs = self.nunknowns()
    #     dofs = self.dofs_of_cells()
    #     coeff = np.asarray(coeff)
    #
    #     rows = dofs.ravel()
    #
    #     if coeff.ndim == 0:
    #         mass = coeff * dV.repeat(dim + 1) / (dim + 1)
    #
    #     elif coeff.shape == (self.mesh.ncells,):
    #         mass = (coeff * dV / (dim + 1)).repeat(dim + 1)
    #
    #     else:
    #         raise ValueError(
    #             "cell-lumped mass expects scalar or cellwise coefficient, "
    #             f"got {coeff.shape=}, {self.mesh.ncells=}"
    #         )
    #
    #     return sparse.coo_matrix(
    #         (mass, (rows, rows)),
    #         shape=(ndofs, ndofs),
    #     ).tocsr()
    # def computeMassMatrixLumped(self, coeff=1):
    #     dim = self.mesh.dimension
    #     dV = self.mesh.geometry.cell_volumes
    #     ndofs = self.nunknowns()
    #     dofs = self.dofs_of_cells()
    #     coeff = np.asarray(coeff)
    #
    #     rows = dofs.ravel()
    #
    #     if coeff.ndim == 0:
    #         mass = coeff * dV.repeat(dim + 1) / (dim + 1)
    #
    #     elif coeff.shape == (self.mesh.ncells,):
    #         mass = (coeff * dV / (dim + 1)).repeat(dim + 1)
    #
    #     elif coeff.shape == (ndofs,):
    #         coeff_loc = coeff[dofs].ravel()
    #         mass = coeff_loc * dV.repeat(dim + 1) / (dim + 1)
    #
    #     else:
    #         raise ValueError(
    #             "lumped mass expects scalar, cellwise, or dofwise coefficient, "
    #             f"got {coeff.shape=}, {self.mesh.ncells=}, {ndofs=}"
    #         )
    #
    #     return sparse.coo_matrix(
    #         (mass, (rows, rows)),
    #         shape=(ndofs, ndofs),
    #     ).tocsr()

    def computeMatrixLps(self, betart, lpsparam=0.1):
        dimension, dV, ndofs, nloc, dofspercell = self.mesh.dimension, self.mesh.geometry.cell_volumes, self.nunknowns(), self.nlocal(), self.dofspercell()
        if not hasattr(self.mesh,'innerfaces'): self.mesh.construct_inner_faces()
        ci = self.mesh.topology.cells_of_inner_faces
        ci0, ci1 = ci[:,0], ci[:,1]
        normalsS = self.mesh.geometry.normals[self.mesh.topology.inner_faces]
        dS = linalg.norm(normalsS, axis=1)
        scale = 0.5*(dV[ci0]+ dV[ci1])
        betan = np.absolute(betart[self.mesh.topology.inner_faces])
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
        dimension, dV, ndofs, nloc, dofspercell = self.mesh.dimension, self.mesh.geometry.cell_volumes, self.nunknowns(), self.nlocal(), self.dofspercell()
        ci = self.mesh.topology.cells_of_inner_faces
        ci0, ci1 = ci[:,0], ci[:,1]
        normalsS = self.mesh.geometry.normals[self.mesh.topology.inner_faces]
        dS = linalg.norm(normalsS, axis=1)
        scale = 0.5*(dV[ci0]+ dV[ci1])
        betan = np.absolute(betart[self.mesh.topology.inner_faces])
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

    def computeEstimator(self, uh, rhs_cell, diffcell=None, skip_faces=None):
        mesh = self.mesh
        dim = mesh.dimension

        uh = np.asarray(uh, dtype=float).reshape(-1)

        if diffcell is None:
            diffcell = np.ones(mesh.ncells)
        else:
            diffcell = np.asarray(diffcell, dtype=float).reshape(-1)

        rhs_cell = np.asarray(rhs_cell, dtype=float).reshape(-1)

        # cellwise flux q_K = A_K grad u_h |_K
        graduh = np.einsum(
            "nij,ni->nj",
            self.cellgrads[:, :, :dim],
            uh[mesh.topology.cells],
        )
        flux = diffcell[:, None] * graduh

        # volume residual
        hK = mesh.geometry.cell_volumes ** (1.0 / dim)
        eta2 = hK ** 2 * rhs_cell ** 2 * mesh.geometry.cell_volumes

        if not hasattr(mesh.topology, "inner_faces"):
            mesh.construct_inner_faces()

        faces = np.asarray(mesh.topology.inner_faces, dtype=int)

        if skip_faces is not None:
            skip_faces = np.asarray(skip_faces, dtype=int)
            faces = faces[~np.isin(faces, skip_faces)]

        ci = mesh.topology.cells_of_faces[faces]
        ok = (ci[:, 0] >= 0) & (ci[:, 1] >= 0)

        faces = faces[ok]
        ci = ci[ok]

        c0 = ci[:, 0]
        c1 = ci[:, 1]

        normalsS = mesh.geometry.normals[faces, :dim]
        dS = np.linalg.norm(normalsS, axis=1)
        nS = normalsS / dS[:, None]

        jump = np.einsum("ij,ij->i", flux[c0] - flux[c1], nS)

        hS = dS if dim == 2 else dS ** (1.0 / (dim - 1.0))
        face_contrib = 0.5 * hS * jump ** 2 * dS

        np.add.at(eta2, c0, face_contrib)
        np.add.at(eta2, c1, face_contrib)

        return float(np.sqrt(np.sum(eta2))), eta2

    def computeFormTransportCellWise(self, du, u, data, type):
        beta = data.betacell
        betart = data.betart

        dim = self.mesh.dimension
        dV = self.mesh.geometry.cell_volumes
        dofspercell = self.dofspercell()
        cellgrads = self.cellgrads[:, :, :dim]

        if type == "centered":
            mus = np.full(dim + 1, 1.0 / (dim + 1))
            mat = np.einsum(
                "n,njk,nk,i,nj->ni",
                dV,
                cellgrads,
                beta,
                mus,
                u[dofspercell],
            )

        elif type == "supg":
            mus = data.md.mus
            mat = np.einsum(
                "n,njk,nk,ni,nj->ni",
                dV,
                cellgrads,
                beta,
                mus,
                u[dofspercell],
            )

        else:
            raise ValueError(f"unknown {type=}")

        np.add.at(du, dofspercell, mat)

        self.massDotBoundary(
            du,
            u,
            coeff=-np.minimum(betart, 0),
            lumped=True,
        )

        return du
    def computeMatrixTransportCellWise(self, data, type):
        beta = data.betacell
        betart = data.betart

        ndofs = self.nunknowns()
        dim = self.mesh.dimension
        dV = self.mesh.geometry.cell_volumes
        cellgrads = self.cellgrads[:, :, :dim]

        if type == "centered":
            mus = np.full(dim + 1, 1.0 / (dim + 1))
            mat = np.einsum(
                "n,njk,nk,i->nij",
                dV,
                cellgrads,
                beta,
                mus,
            )

        elif type == "supg":
            mus = data.md.mus
            mat = np.einsum(
                "n,njk,nk,ni->nij",
                dV,
                cellgrads,
                beta,
                mus,
            )

        else:
            raise ValueError(f"unknown {type=}")

        A = sparse.coo_matrix(
            (mat.ravel(), (self.rows, self.cols)),
            shape=(ndofs, ndofs),
        ).tocsr()

        A -= self.computeBdryMassMatrix(
            coeff=np.minimum(betart, 0),
            lumped=True,
        )

        return A
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
