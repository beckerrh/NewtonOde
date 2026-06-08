# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
from . import barycentric, p1general, data, mesh_transfer

#=================================================================#
class P1(p1general.P1general):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setMesh(self, mesh):
        super().setMesh(mesh)
        # P1
        self.computeStencilCell(self.mesh.topology.cells)
        self.node_x, self.node_y, self.node_z = self._xyz_from_points(
            self.mesh.geometry.points
        )
    def nlocal(self): return self.mesh.dimension+1
    def nunknowns(self): return self.mesh.nnodes
    def dofspercell(self): return self.mesh.topology.cells
    def computeCellGrads(self):
        ncells, normals, cells_of_faces, faces_of_cells, dV = self.mesh.ncells, self.mesh.geometry.normals, self.mesh.topology.cells_of_faces, self.mesh.topology.faces_of_cells, self.mesh.geometry.cell_volumes
        scale = -1/self.mesh.dimension
        return scale*(normals[faces_of_cells].T * self.mesh.sigma.T / dV.T).T

    def dofs_of_cells(self):
        return self.mesh.topology.cells

    def masslocal(self):
        return barycentric.tensor(d=self.mesh.dimension, k=2)

    def masslocal_variable(self):
        return barycentric.tensor(d=self.mesh.dimension, k=3)

    def tonode(self, u): return u
    def to_p1(self, u): return u

    def dof_coordinates_xyz(self):
        return self.node_x, self.node_y, self.node_z

    def dirichlet_dofs(self, bdrydata=None):
        if bdrydata is None:
            bdrydata = self.bdrydata

        faces = self.boundary_faces_from_bdrydata(bdrydata)
        nodes = np.unique(self.mesh.topology.faces[faces].ravel())
        return nodes.astype(int)

    def prepareBoundary(self, colorsdir, colorsflux=[]):
        bdrydata = data.BdryData()
        bdrydata.nodesdir={}
        bdrydata.nodedirall = np.empty(shape=(0), dtype=self.mesh.topology.faces.dtype)
        for color in colorsdir:
            facesdir = self.mesh.labels.boundary[color]
            bdrydata.nodesdir[color] = np.unique(self.mesh.topology.faces[facesdir].flat[:])
            bdrydata.nodedirall = np.unique(np.union1d(bdrydata.nodedirall, bdrydata.nodesdir[color]))
        bdrydata.nodesinner = np.setdiff1d(np.arange(self.mesh.nnodes, dtype=self.mesh.topology.faces.dtype),bdrydata.nodedirall)
        bdrydata.nodesdirflux={}
        for color in colorsflux:
            facesdir = self.mesh.labels.boundary[color]
            bdrydata.nodesdirflux[color] = np.unique(self.mesh.topology.faces[facesdir].ravel())
        return bdrydata

    def matrixBoundaryStrong(self, A, bdrydata):
        n = self.nunknowns()
        bdofs = bdrydata.nodedirall

        mask = np.ones(n)
        mask[bdofs] = 0.0
        D = sparse.diags(mask, format="csr")

        A = D @ A

        A = A.tolil()
        A[bdofs, bdofs] = 1.0
        return A.tocsr()
    def vectorBoundaryStrong(self, b, bdrycond, bdrydata, method="strong"):
        # method = self.params_str['dirichletmethod']
        # if method not in ['strong','new']: return
        nodesdir, nodedirall, nodesinner, nodesdirflux = bdrydata.nodesdir, bdrydata.nodedirall, bdrydata.nodesinner, bdrydata.nodesdirflux
        x, y, z = self.mesh.geometry.points.T
        for color, nodes in nodesdirflux.items():
            bdrydata.bsaved[color] = b[nodes]
        if method == 'strong':
            for color, nodes in nodesdir.items():
                if color in bdrycond.fct:
                    dirichlet = bdrycond.fct[color](x[nodes], y[nodes], z[nodes])
                    b[nodes] = dirichlet
                else:
                    b[nodes] = 0
            # b[nodesinner] -= bdrydata.A_inner_dir * b[nodedirall]
        else:
            help = np.zeros_like(b)
            for color, nodes in nodesdir.items():
                if color in bdrycond.fct:
                    dirichlet = bdrycond.fct[color](x[nodes], y[nodes], z[nodes])
                    help[nodes] = dirichlet
            # b[nodesinner] -= bdrydata.A_inner_dir * help[nodedirall]
            b[nodedirall] = bdrydata.A_dir_dir * help[nodedirall]
        return b
    def vectorBoundaryStrongEqual(self, du, u, bdrydata):
        # if self.params_str['dirichletmethod']=="nitsche": return
        nodedirall = bdrydata.nodedirall
        du[nodedirall] = u[nodedirall]
    def vectorBoundaryStrongZero(self, du, bdrydata, method="strong"):
        # if self.params_str['dirichletmethod']=="nitsche": return
        du[bdrydata.nodedirall] = 0
    def computeRhsNitscheDiffusion(self, nitsche_param, b, diffcoff, colors, udir=None, bdrycondfct=None, coeff=1, lumped=False):
        if udir is None:
            udir = self.interpolateBoundary(colors, bdrycondfct)
        assert udir.shape[0]==self.mesh.nnodes
        dim  = self.mesh.dimension
        massloc = barycentric.tensor(d=dim - 1, k=2)
        massloc = np.diag(np.sum(massloc,axis=1))
        faces = self.mesh.bdryFaces(colors)
        nodes, cells, normalsS = self.mesh.topology.faces[faces], self.mesh.topology.cells_of_faces[faces,0], self.mesh.geometry.normals[faces,:dim]
        dS = linalg.norm(normalsS, axis=1)
        simp, dV = self.mesh.topology.cells[cells], self.mesh.geometry.cell_volumes[cells]
        dS *= nitsche_param * coeff * diffcoff[cells] * dS / dV
        r = np.einsum('n,kl,nl->nk', dS, massloc, udir[nodes])
        np.add.at(b, nodes, r)
        cellgrads = self.cellgrads[cells, :, :dim]
        u = udir[nodes].mean(axis=1)
        mat = np.einsum('f,fk,fik->fi', coeff*u*diffcoff[cells], normalsS, cellgrads)
        np.add.at(b, simp, -mat)
    def computeFormNitscheDiffusion(self, nitsche_param, du, u, diffcoff, colorsdir, lumped=False):
        assert u.shape[0]==self.mesh.nnodes
        dim  = self.mesh.dimension
        massloc = barycentric.tensor(d=dim - 1, k=2)
        massloc = np.diag(np.sum(massloc,axis=1))
        faces = self.mesh.bdryFaces(colorsdir)
        nodes, cells, normalsS = self.mesh.topology.faces[faces], self.mesh.topology.cells_of_faces[faces,0], self.mesh.geometry.normals[faces,:dim]
        dS = linalg.norm(normalsS, axis=1)
        simp, dV = self.mesh.topology.cells[
cells], self.mesh.geometry.cell_volumes[cells]
        dS *= nitsche_param * diffcoff[cells] * dS / dV
        r = np.einsum('n,kl,nl->nk', dS, massloc, u[nodes])
        np.add.at(du, nodes, r)
        cellgrads = self.cellgrads[cells, :, :dim]
        um = u[nodes].mean(axis=1)
        mat = np.einsum('f,fk,fik->fi', um*diffcoff[cells], normalsS, cellgrads)
        np.add.at(du, simp, -mat)
        mat = np.einsum('f,fk,fjk,fj->f', diffcoff[cells]/dim, normalsS, cellgrads,u[simp]).repeat(dim).reshape(faces.shape[0],dim)
        np.add.at(du, nodes, -mat)
    def computeMatrixNitscheDiffusion(self, nitsche_param, diffcoff, colors, coeff=1, lumped=False):
        nnodes, ncells, dim, nlocal  = self.mesh.nnodes, self.mesh.ncells, self.mesh.dimension, self.nlocal()
        faces = self.mesh.bdryFaces(colors)
        cells = self.mesh.topology.cells_of_faces[faces, 0]
        normalsS = self.mesh.geometry.normals[faces, :dim]
        dS = np.linalg.norm(normalsS, axis=1)
        dV = self.mesh.geometry.cell_volumes[cells]
        cellgrads = self.cellgrads[cells, :, :dim]
        simp = self.mesh.topology.cells[
cells]
        facenodes = self.mesh.topology.faces[faces]
        cols = np.tile(simp,dim)
        rows = facenodes.repeat(dim+1)
        mat = np.einsum('f,fk,fjk,i->fij', diffcoff[cells]/dim, normalsS, cellgrads, np.ones(dim))
        # mat = np.repeat(mat,dim)
        # print(f"{cols.shape=} {rows.shape=} {mat.shape=}")
        AN = sparse.coo_matrix((mat.ravel(), (rows.ravel(), cols.ravel())), shape=(nnodes, nnodes)).tocsr()
        massloc = barycentric.tensor(d=dim - 1, k=2)
        massloc = np.diag(np.sum(massloc,axis=1))
        # print(f"{massloc=}")
        mat = np.einsum('f,ij->fij', nitsche_param * dS**2/dV*diffcoff[cells], massloc)
        # mat = np.repeat(coeff * diffcoff[cells]/dS, dim)
        rows = np.repeat(facenodes,dim)
        cols = np.tile(facenodes,dim)
        AD = sparse.coo_matrix((mat.ravel(), (rows.ravel(), cols.ravel())), shape=(nnodes, nnodes)).tocsr()
        return  - AN - AN.T + AD
    def computeBdryNormalFluxNitsche(self, nitsche_param, u, colors, udir, diffcoff):
        flux= np.zeros(len(colors))
        nnodes, dim  = self.mesh.nnodes, self.mesh.dimension
        massloc = barycentric.tensor(d=dim - 1, k=2)
        massloc = np.diag(np.sum(massloc,axis=1))
        for i,color in enumerate(colors):
            faces = self.mesh.labels.boundary[color]
            normalsS = self.mesh.geometry.normals[faces,:dim]
            dS = linalg.norm(normalsS, axis=1)
            nodes = self.mesh.topology.faces[faces]
            cells = self.mesh.topology.cells_of_faces[faces,0]
            simp = self.mesh.topology.cells[
cells]
            cellgrads = self.cellgrads[cells, :, :dim]
            dV = self.mesh.geometry.cell_volumes[cells]
            flux[i] = np.einsum('nj,n,ni,nji->', u[simp], diffcoff[cells], normalsS, cellgrads)
            uD = u[nodes]-udir[nodes]
            dV = self.mesh.geometry.cell_volumes[cells]
            flux[i] -= np.einsum('n,kl,nl->', nitsche_param * diffcoff[cells] * dS**2 / dV, massloc, uD)
            # flux[i] /= np.sum(dS)
        return flux
    # interpolate
    def interpolate(self, f):
        x, y, z = self.mesh.geometry.points.T
        return f(x, y, z)
    def interpolateBoundary(self, colors, f, lumped=False):
        """
        :param colors: set of colors to interpolate
        :param f: ditct of functions
        :return:
        """
        b = np.zeros(self.mesh.nnodes)
        # print(f"{type(f)=} {colors=} {len(f)=}")
        # if len(f) < len(colors):
        #     raise ValueError(f"{type(f)=} {colors=} {len(f)=}")
        for color in colors:
            if not color in f or not f[color]: continue
            faces = self.mesh.labels.boundary[color]
            normalsS = self.mesh.geometry.normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS[:,np.newaxis]
            nx, ny, nz = normalsS.T
            nodes = np.unique(self.mesh.topology.faces[faces].reshape(-1))
            x, y, z = self.mesh.geometry.points[nodes].T
            # constant normal !!
            nx, ny, nz = np.mean(normalsS, axis=0)
            try:
                b[nodes] = f[color](x, y, z, nx, ny, nz)
            except:
                b[nodes] = f[color](x, y, z)
        return b
    def computeBdryMassMatrix(self, colors=None, coeff=1, lumped=False):
        nnodes = self.mesh.nnodes
        rows = np.empty(shape=(0), dtype=int)
        cols = np.empty(shape=(0), dtype=int)
        mat = np.empty(shape=(0), dtype=float)
        if colors is None: colors = self.mesh.labels.boundary.keys()
        for color in colors:
            faces = self.mesh.labels.boundary[color]
            normalsS = self.mesh.geometry.normals[faces]
            if isinstance(coeff, dict):
                dS = linalg.norm(normalsS, axis=1)*coeff[color]
            else:
                dS = linalg.norm(normalsS, axis=1)*coeff[faces]
            nodes = self.mesh.topology.faces[faces]
            if lumped:
                dS /= self.mesh.dimension
                rows = np.append(rows, nodes)
                cols = np.append(cols, nodes)
                mass = np.repeat(dS, self.mesh.dimension)
                mat = np.append(mat, mass)
            else:
                nloc = self.mesh.dimension
                rows = np.append(rows, np.repeat(nodes, nloc).ravel())
                cols = np.append(cols, np.tile(nodes, nloc).ravel())
                massloc = barycentric.tensor(d=self.mesh.dimension-1, k=2)
                mat = np.append(mat, np.einsum('n,kl->nkl', dS, massloc).ravel())
        return sparse.coo_matrix((mat, (rows, cols)), shape=(nnodes, nnodes)).tocsr()
    def computeMatrixTransportSupg(self, data, method):
        return self.computeMatrixTransportCellWise(data, type='supg')
    def computeMatrixTransportLps(self, data):
        A = self.computeMatrixTransportCellWise(data, type='centered')
        A += self.computeMatrixLps(data.betart)
        return A
    def computeMassMatrixSupg(self, xd, data, coeff=1):
        dim, dV, nnodes, xK = self.mesh.dimension, self.mesh.geometry.cell_volumes, self.mesh.nnodes, self.mesh.geometry.cell_centers
        massloc = barycentric.tensor(d=dim, k=2)
        mass = np.einsum('n,ij->nij', coeff*dV, massloc)
        massloc = barycentric.tensor(d=dim, k=1)
        # marche si xd = xK + delta*betaC
        # mass += np.einsum('n,nik,nk,j -> nij', coeff*delta*dV, self.cellgrads[:,:,:dim], betaC, massloc)
        mass += np.einsum('n,nik,nk,j -> nij', coeff*dV, self.cellgrads[:,:,:dim], xd[:,:dim]-xK[:,:dim], massloc)
        return sparse.coo_matrix((mass.ravel(), (self.rows, self.cols)), shape=(nnodes, nnodes)).tocsr()
    # dotmat
    def formDiffusion(self, du, u, coeff):
        graduh = np.einsum('nij,ni->nj', self.cellgrads, u[self.mesh.topology.cells])
        graduh = np.einsum('ni,n->ni', graduh, self.mesh.geometry.cell_volumes*coeff)
        # du += np.einsum('nj,nij->ni', graduh, self.cellgrads)
        raise ValueError(f"graduh {graduh.shape} {du.shape}")
        return du
    def massDotCell(self, b, f, coeff=1):
        assert f.shape[0] == self.mesh.ncells
        dimension, simplices, dV = self.mesh.dimension, self.mesh.topology.cells, self.mesh.geometry.cell_volumes
        massloc = 1/(dimension+1)
        np.add.at(b, simplices, (massloc*coeff*dV*f)[:, np.newaxis])
        return b
    def massDot(self, b, f, coeff=1):
        dim, simplices, dV = self.mesh.dimension, self.mesh.topology.cells, self.mesh.geometry.cell_volumes
        massloc = barycentric.tensor(d=dim, k=2)
        r = np.einsum('n,kl,nl->nk', coeff * dV, massloc, f[simplices])
        np.add.at(b, simplices, r)
        return b
    def massDotSupg(self, b, f, data, coeff=1):
        if self.params_str['convmethod'][:4] != 'supg': return
        dim, simplices, dV = self.mesh.dimension, self.mesh.topology.cells, self.mesh.geometry.cell_volumes
        r = np.einsum('n,nk,n->nk', coeff*dV, data.md.mus-1/(dim+1), f[simplices].mean(axis=1))
        np.add.at(b, simplices, r)
        return b
    def massDotBoundary(self, b=None, f=None, colors=None, coeff=1, lumped=False):
        if colors is None: colors = self.mesh.labels.boundary.keys()
        for color in colors:
            faces = self.mesh.labels.boundary[color]
            normalsS = self.mesh.geometry.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            nodes = self.mesh.topology.faces[faces]
            if isinstance(coeff, (int,float)): dS *= coeff
            elif isinstance(coeff, dict): dS *= coeff[color]
            else:
                # print(f"{coeff.shape=} {self.mesh.nfaces=}")
                assert coeff.shape[0]==self.mesh.nfaces
                dS *= coeff[faces]
            # print(f"{scalemass=}")
            if lumped:
                np.add.at(b, nodes, f[nodes]*dS[:,np.newaxis]/self.mesh.dimension)
            else:
                massloc = barycentric.tensor(d=self.mesh.dimension - 1, k=2)
                r = np.einsum('n,kl,nl->nk', dS, massloc, f[nodes])
                np.add.at(b, nodes, r)
        return b
    # rhs
    def computeRhsMass(self, b, rhs, mass):
        if rhs is None: return b
        x, y, z = self.mesh.geometry.points.T
        b += mass * rhs(x, y, z)
        return b
    def computeRhsCell(self, b, rhscell):
        if rhscell is None: return b
        if isinstance(rhscell,dict):
            assert set(rhscell.keys())==set(self.mesh.labels.cell.keys())
            scale = 1 / (self.mesh.dimension + 1)
            for label, fct in rhscell.items():
                if fct is None: continue
                cells = self.mesh.labels.cell[label]
                xc, yc, zc = self.mesh.geometry.cell_centers[cells].T
                bC = scale * fct(xc, yc, zc) * self.mesh.geometry.cell_volumes[cells]
                # print("bC", bC)
                np.add.at(b, self.mesh.topology.cells[
cells].T, bC)
        else:
            fp1 = self.interpolateCell(rhscell)
            self.massDotCell(b, fp1, coeff=1)
        return b
    def computeRhsPoint(self, b, rhspoint):
        if rhspoint is None: return b
        for label, fct in rhspoint.items():
            if fct is None: continue
            points = self.mesh.labels.vertex[label]
            xc, yc, zc = self.mesh.geometry.points[points].T
            # print("xc, yc, zc, f", xc, yc, zc, fct(xc, yc, zc))
            b[points] += fct(xc, yc, zc)
        return b
    def computeRhsBoundary(self, b, bdryfct, colors):
        normals =  self.mesh.geometry.normals
        scale = 1 / self.mesh.dimension
        for color in colors:
            faces = self.mesh.labels.boundary[color]
            if not color in bdryfct or bdryfct[color] is None: continue
            normalsS = normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS[:,np.newaxis]
            assert(dS.shape[0] == len(faces))
            xf, yf, zf = self.mesh.geometry.face_centers[faces].T
            nx, ny, nz = normalsS.T
            bS = scale * bdryfct[color](xf, yf, zf, nx, ny, nz) * dS
            np.add.at(b, self.mesh.topology.faces[faces].T, bS)
        return b
    def computeRhsBoundaryMass(self, b, bdrycond, types, mass):
        normals =  self.mesh.geometry.normals
        help = np.zeros(self.mesh.nnodes)
        for color, faces in self.mesh.labels.boundary.items():
            if bdrycond.type[color] not in types: continue
            if not color in bdrycond.fct or bdrycond.fct[color] is None: continue
            normalsS = normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            normalsS = normalsS/dS[:,np.newaxis]
            nx, ny, nz = normalsS.T
            assert(dS.shape[0] == len(faces))
            nodes = np.unique(self.mesh.topology.faces[faces].reshape(-1))
            x, y, z = self.mesh.geometry.points[nodes].T
            # constant normal !!
            nx, ny, nz = np.mean(normalsS, axis=0)
            help[nodes] = bdrycond.fct[color](x, y, z, nx, ny, nz)
        # print("help", help)
        b += mass*help
        return b
    # postprocess
    def to_cell(self, u):
        return np.mean(u[self.mesh.topology.cells], axis=1)
    def computeBdryMean(self, u, colors):
        mean, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.labels.boundary[color]
            normalsS = self.mesh.geometry.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            mean[i] = np.sum(dS*np.mean(u[self.mesh.topology.faces[faces]],axis=1))
        return mean/omega
    def comuteFluxOnRobin(self, u, faces, dS, uR, cR):
        uhmean =  np.sum(dS * np.mean(u[self.mesh.topology.faces[faces]], axis=1))
        xf, yf, zf = self.mesh.geometry.face_centers[faces].T
        nx, ny, nz = np.mean(self.mesh.geometry.normals[faces], axis=0)
        if uR:
            try:
                uRmean =  np.sum(dS * uR(xf, yf, zf, nx, ny, nz))
            except:
                uRmean =  np.sum(dS * uR(xf, yf, zf))
        else: uRmean=0
        return cR*(uRmean-uhmean)
    def computeBdryNormalFlux(self, u, colors, bdrydata, bdrycond, diffcoff):
        flux, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.labels.boundary[color]
            normalsS = self.mesh.geometry.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            if color in bdrydata.bsaved.keys():
                bs, As = bdrydata.bsaved[color], bdrydata.Asaved[color]
                flux[i] = np.sum(As * u - bs)
            else:
                flux[i] = self.comuteFluxOnRobin(u, faces, dS, bdrycond.fct[color], bdrycond.param[color])
        return flux
    def computeBdryFct(self, u, colors):
        nodes = np.empty(shape=(0), dtype=int)
        for color in colors:
            faces = self.mesh.labels.boundary[color]
            nodes = np.unique(np.union1d(nodes, self.mesh.topology.faces[faces].ravel()))
        return self.mesh.geometry.points[nodes], u[nodes]
    def computePointValues(self, u, colors):
        up = np.empty(len(colors))
        for i,color in enumerate(colors):
            nodes = self.mesh.labels.vertex[color]
            # print(f"{nodes=} {self.mesh.geometry.points[nodes]=}")
            up[i] = u[nodes]
        return up
    def computeLineValues(self, u, colors):
        raise NotImplementedError()
        up = np.empty(len(colors))
        for i,color in enumerate(colors):
            lines = self.mesh.labels.line[color]
            nodes = np.unique(lines)
            print(f"{u[nodes]=}")
            print(f"{self.mesh.geometry.points[nodes,0]=}")
            print(f"{self.mesh.geometry.points[nodes,1]=}")
            import matplotlib.pyplot as plt
            plt.plot(self.mesh.geometry.points[nodes,0], u[nodes])
            plt.show()
            # print(f"{np.unique(lines)=}")
            # print(f"{self.mesh.geometry.points[lines]=}")
        return up
    def computeMeanValues(self, u, colors):
        up = np.empty(len(colors))
        for i, color in enumerate(colors):
            cells = self.mesh.labels.cell[color]
            up[i] = np.sum(np.mean(u[self.mesh.topology.cells[
cells]],axis=1)*self.mesh.geometry.cell_volumes[cells])
        return up

    def build_scalar_prolongation_to_refined_mesh(self, info):
        return mesh_transfer.p1_prolongation(info)

    # P1
    def cellmean_vector(self, U):
        cells = self.mesh.topology.cells
        return np.mean(U[:, cells], axis=2)

# ------------------------------------- #
if __name__ == '__main__':
    from FEM2D.meshes_new import testmeshes
    from FEM2D.meshes_new import plotmesh
    import matplotlib.pyplot as plt
    mesh = testmeshes.backwardfacingstep(h=0.2)
    fem = P1(mesh=mesh)
    print(f"{fem=}")
    u = fem.test()
    plotmesh.meshWithBoundaries(mesh)
    plotmesh.meshWithData(mesh, point_data={'u':u}, title="P1 Test", alpha=1)
    plt.show()
