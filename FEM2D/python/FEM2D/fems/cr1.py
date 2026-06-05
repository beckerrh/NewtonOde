# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
from . import barycentric, p1general, data, mesh_transfer, rt0

@staticmethod
def faces_of_cells_not_on_faces(faces_of_cells, cells0, cells1, faces):
    f0 = faces_of_cells[cells0]
    f1 = faces_of_cells[cells1]

    mask0 = f0 != faces[:, None]
    mask1 = f1 != faces[:, None]

    return (
        f0[mask0].reshape(-1, f0.shape[1] - 1),
        f1[mask1].reshape(-1, f1.shape[1] - 1),
    )

#=================================================================#
class CR1(p1general.P1general):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.computeStencilCell(self.mesh.topology.faces_of_cells)
        self.cellgrads = self.computeCellGrads()
        self.dim = mesh.dimension
        if self.dim == 2:
            self.face_x = mesh.geometry.face_centers[:, 0]
            self.face_y = mesh.geometry.face_centers[:, 1]
            self.face_z = np.zeros(mesh.nfaces)
        elif self.dim == 3:
            self.face_x = mesh.geometry.face_centers[:, 0]
            self.face_y = mesh.geometry.face_centers[:, 1]
            self.face_z = mesh.geometry.face_centers[:, 2]
        else:
            raise ValueError(f"Unsupported dimension {self.dim}")
        if self.dim == 2:
            self.cell_x = mesh.geometry.cell_centers[:, 0]
            self.cell_y = mesh.geometry.cell_centers[:, 1]
            self.cell_z = np.zeros(mesh.ncells)
        else:
            self.cell_x = mesh.geometry.cell_centers[:, 0]
            self.cell_y = mesh.geometry.cell_centers[:, 1]
            self.cell_z = mesh.geometry.cell_centers[:, 2]

    def nlocal(self): return self.mesh.dimension+1
    def nunknowns(self): return self.mesh.nfaces
    def dofspercell(self): return self.mesh.topology.faces_of_cells
    def tonode(self, u):
        # print(f"{u=}")
        if u.shape[0] != self.mesh.nfaces: raise ValueError(f"{u.shape=} {self.mesh.nfaces=}")
        unodes = np.zeros(self.mesh.nnodes, dtype=u.dtype)
        scale = self.mesh.dimension
        np.add.at(unodes, self.mesh.topology.cells.T, np.sum(u[self.mesh.topology.faces_of_cells], axis=1)[np.newaxis,:])
        np.add.at(unodes, self.mesh.topology.cells.T, -scale*u[self.mesh.topology.faces_of_cells].T)
        countnodes = np.zeros(self.mesh.nnodes, dtype=int)
        np.add.at(countnodes, self.mesh.topology.cells.T, 1)
        unodes /= countnodes
        return unodes
    # def prepareAdvection(self, beta, scale):
    #     method = self.params_str['convmethod']
    #     rt = FEM2D.fems.rt0.RT0(mesh=self.mesh)
    #     betart = scale*rt.interpolate(beta)
    #     beta = rt.toCell(betart)
    #     convdata = FEM2D.fems.data.ConvectionData(beta=beta, betart=betart)
    #     dim = self.mesh.dimension
    #     self.mesh.construct_inner_faces()
    #     if method == 'upwalg' or method == 'lps':
    #          return convdata 
    #     elif method == 'supg':
    #         md = move.move_midpoints(self.mesh, beta, bound=1/dim)
    #         # self.md = move.move_midpoints(self.mesh, beta, candidates='all')
    #         # self.md.plot(self.mesh, beta, type='midpoints')
    #     elif method == 'supg2':
    #         md = move.move_midpoint_to_neighbour(self.mesh, betart)
    #         # self.md = move.move_midpoints(self.mesh, beta, candidates='all')
    #         # self.md = move.move_midpoints(self.mesh, beta, candidates='all')
    #         # print(f"{self.md.mus=}")
    #         # self.md.plot(self.mesh, beta, type='midpoints')
    #     elif method == 'upw':
    #         md = move.move_midpoints(self.mesh, beta, bound=1/dim)
    #         # self.md = move.move_midpoint_to_neighbour(self.mesh, betart)
    #         # self.md = move.move_midpoints(self.mesh, -beta, bound=1/d)
    #         # self.md = move.move_midpoints(self.mesh, -beta, candidates='all')
    #         # self.md.plot(self.mesh, beta, type='midpoints')
    #     elif method == 'upw2':
    #         md = move.move_midpoints(self.mesh, -beta, bound=1/dim)
    #     else:
    #         raise ValueError(f"don't know {method=}")
    #     convdata.md = md
    #     return convdata
    def computeCellGrads(self):
        normals, faces_of_cells, dV = self.mesh.geometry.normals, self.mesh.topology.faces_of_cells, self.mesh.geometry.cell_volumes
        return (normals[faces_of_cells].T * self.mesh.sigma.T / dV.T).T
    # strong bc
    def prepareBoundary(self, colorsdir, colorsflux=[]):
        bdrydata = data.BdryData()
        bdrydata.facesdirall = np.empty(shape=(0), dtype=np.uint32)
        bdrydata.colorsdir = colorsdir
        for color in colorsdir:
            facesdir = self.mesh.labels.boundary[color]
            bdrydata.facesdirall = np.unique(np.union1d(bdrydata.facesdirall, facesdir))
        bdrydata.facesinner = np.setdiff1d(np.arange(self.mesh.nfaces, dtype=int), bdrydata.facesdirall)
        bdrydata.facesdirflux = {}
        for color in colorsflux:
            bdrydata.facesdirflux[color] = self.mesh.labels.boundary[color]
        return bdrydata
    def computeRhsNitscheDiffusion(self, nitsche_param, b, diffcoff, colors, udir=None, bdrycondfct=None, coeff=1, lumped=False):
        if udir is None:
            udir = self.interpolateBoundary(colors, bdrycondfct)
        if not udir.shape[0] == self.mesh.nfaces:
            raise ValueError(f"{udir.shape[0]=} {self.mesh.nfaces=}")
        dim, faces = self.mesh.dimension, self.mesh.bdryFaces(colors)
        cells = self.mesh.topology.cells_of_faces[faces,0]
        normalsS = self.mesh.geometry.normals[faces][:,:dim]

        dS, dV = np.linalg.norm(normalsS,axis=1), self.mesh.geometry.cell_volumes[cells]
        mat = np.einsum('f,fi,fji->fj', coeff*udir[faces]*diffcoff[cells], normalsS, self.cellgrads[cells, :, :dim])
        np.add.at(b, self.mesh.topology.faces_of_cells[cells], -mat)
        self.massDotBoundary(b, f=udir, colors=colors, coeff=coeff*nitsche_param*diffcoff[cells] * dS/dV, lumped=lumped)
    def computeFormNitscheDiffusion(self, nitsche_param, du, u, diffcoff, colorsdir, lumped=False):
        assert u.shape[0] == self.mesh.nfaces

        dim = self.mesh.dimension
        faces = self.mesh.bdryFaces(colorsdir)
        cells = self.mesh.topology.cells_of_faces[faces, 0]

        foc = self.mesh.topology.faces_of_cells[cells]
        normalsS = self.mesh.geometry.normals[faces][:, :dim]
        cellgrads = self.cellgrads[cells, :, :dim]

        dS = np.linalg.norm(normalsS, axis=1)
        dV = self.mesh.geometry.cell_volumes[cells]

        # - AN @ u
        mat = np.einsum(
            "f,fk,fjk,fj->f",
            diffcoff[cells],
            normalsS,
            cellgrads,
            u[foc],
        )
        np.add.at(du, faces, -mat)

        # - AN.T @ u
        mat = np.einsum(
            "f,fk,fik->fi",
            diffcoff[cells] * u[faces],
            normalsS,
            cellgrads,
        )
        np.add.at(du, foc, -mat)

        # + AD @ u
        self.massDotBoundary(
            du,
            f=u,
            colors=colorsdir,
            coeff=nitsche_param * diffcoff[cells] * dS / dV,
            lumped=lumped,
        )
    def computeMatrixNitscheDiffusion(self, nitsche_param, diffcoff, colors, coeff=1, lumped=False):
        nfaces, ncells, dim, nlocal  = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension, self.nlocal()
        # if self.params_str['dirichletmethod'] != 'nitsche': return sparse.coo_matrix((nfaces,nfaces))
        # nitsche_param=self.params_float['nitscheparam']
        faces = self.mesh.bdryFaces(colors)
        if not isinstance(coeff, (float,int)): assert coeff.shape[0]==faces.shape[0]
        cells = self.mesh.topology.cells_of_faces[faces, 0]
        normalsS = self.mesh.geometry.normals[faces][:, :dim]

        cols = self.mesh.topology.faces_of_cells[cells, :].ravel()
        rows = faces.repeat(nlocal)
        mat = np.einsum('f,fi,fji->fj', coeff*diffcoff[cells], normalsS, self.cellgrads[cells, :, :dim]).ravel()
        AN = sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces)).tocsr()
        # AD = sparse.diags(AN.diagonal(), offsets=(0), shape=(nfaces, nfaces))
        dS = np.linalg.norm(normalsS,axis=1)
        dV = self.mesh.geometry.cell_volumes[cells]
        # AD = sparse.coo_matrix((dS**2/dV,(faces,faces)), shape=(nfaces, nfaces))
        AD = self.computeBdryMassMatrix(colors=colors, coeff=coeff*diffcoff[cells]*nitsche_param*dS/dV, lumped=lumped)
        return AD - AN - AN.T
    def computeBdryNormalFluxNitsche(self, nitsche_param, u, colors, udir, diffcoff):
        # nitsche_param=self.params_float['nitscheparam']
        #TODO correct flux computation Nitsche
        flux= np.zeros(len(colors))
        nfaces, ncells, dim, nlocal  = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension, self.nlocal()
        faces_of_cells = self.mesh.topology.faces_of_cells
        for i,color in enumerate(colors):
            faces = self.mesh.labels.boundary[color]
            cells = self.mesh.topology.cells_of_faces[faces, 0]
            normalsS = self.mesh.geometry.normals[faces,:dim]
            cellgrads = self.cellgrads[cells, :, :dim]
            foc = faces_of_cells[cells]
            flux[i] = np.einsum('fj,f,fi,fji->', u[foc], diffcoff[cells], normalsS, cellgrads)
            dS = np.linalg.norm(normalsS,axis=1)
            dV = self.mesh.geometry.cell_volumes[cells]
            flux[i] -= self.massDotBoundary(b=None, f=u-udir, colors=[color], coeff=nitsche_param * diffcoff[cells]*dS/dV)
            # flux[i] /= np.sum(dS)
        return flux
    def vectorBoundaryStrongEqual(self, du, u, bdrydata):
        # if self.params_str['dirichletmethod']=="nitsche": return
        facesdirall = bdrydata.facesdirall
        du[facesdirall] = u[facesdirall]
    def vectorBoundaryStrongZero(self, du, bdrydata):
        # if self.params_str['dirichletmethod']=="nitsche": return
        du[bdrydata.facesdirall] = 0
    def vectorBoundaryStrong(self, b, bdrycond, bdrydata):
        # method = self.params_str['dirichletmethod']
        # if method not in ['strong','new']: return
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        x, y, z = self.face_x, self.face_y, self.face_z
        for color, faces in facesdirflux.items():
            bdrydata.bsaved[color] = b[faces]
        help = np.zeros_like(b)
        for color in colorsdir:
            faces = self.mesh.labels.boundary[color]
            if color in bdrycond.fct:
                dirichlet = bdrycond.fct[color]
                help[faces] = dirichlet(x[faces], y[faces], z[faces])
        # b[facesinner] -= bdrydata.A_inner_dir * help[facesdirall]
        # if method == 'strong':
        b[facesdirall] = help[facesdirall]
        # else:
        #     b[facesdirall] = bdrydata.A_dir_dir * help[facesdirall]
    def matrixBoundaryStrong(self, A, bdrydata, method='strong'):
        # method = self.params_str['dirichletmethod']
        # if method not in ['strong','new']: return
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        nfaces = self.mesh.nfaces
        for color, faces in facesdirflux.items():
            nb = faces.shape[0]
            help = sparse.dok_matrix((nb, nfaces))
            for i in range(nb): help[i, faces[i]] = 1
            bdrydata.Asaved[color] = help.dot(A)
        bdrydata.A_inner_dir = A[facesinner, :][:, facesdirall]
        help = np.ones((nfaces))
        help[facesdirall] = 0
        help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
        # A = help.dot(A.dot(help))
        diag = np.zeros((nfaces))
        if method == 'strong':
            diag[facesdirall] = 1.0
            diag = sparse.dia_matrix((diag, 0), shape=(nfaces, nfaces))
        else:
            bdrydata.A_dir_dir = self.dirichlet_strong*A[facesdirall, :][:, facesdirall]
            diag[facesdirall] = np.sqrt(self.dirichlet_strong)
            diag = sparse.dia_matrix((diag, 0), shape=(nfaces, nfaces))
            diag = diag.dot(A.dot(diag))
        A = help.dot(A)
        A += diag
        return A
    # interpolate
    def interpolate(self, f):
        return f(self.face_x, self.face_y, self.face_z)
    def interpolateBoundary(self, colors, f, lumped=False):
        """
        :param colors: set of colors to interpolate
        :param f: ditct of functions
        :return:
        """
        b = np.zeros(self.mesh.nfaces)
        if lumped:
            for color in colors:
                if not color in f or not f[color]: continue
                faces = self.mesh.labels.boundary[color]
                ci = self.mesh.topology.cells_of_faces[faces][:, 0]
                foc = self.mesh.topology.faces_of_cells[ci]
                x = self.face_x[foc]
                y = self.face_y[foc]
                z = self.face_z[foc]
                mask = foc != faces[:, np.newaxis]
                fi = foc[mask].reshape(foc.shape[0], foc.shape[1] - 1)
                normalsS = self.mesh.geometry.normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                normalsS = normalsS/dS[:,np.newaxis]
                nx, ny, nz = normalsS.T
                # x, y, z = self.mesh.geometry.face_centers[faces].T
                try:
                    b[faces] = f[color](x, y, z, nx, ny, nz)
                except:
                    b[faces] = f[color](x, y, z)
            return b
        for color in colors:
            if not color in f or not f[color]: continue
            faces = self.mesh.labels.boundary[color]
            normalsS = self.mesh.geometry.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            normalsS = normalsS / dS[:, np.newaxis]
            nx, ny, nz = normalsS.T
            ci = self.mesh.topology.cells_of_faces[faces][:, 0]
            foc = self.mesh.topology.faces_of_cells[ci]
            x = self.face_x[foc]
            y = self.face_y[foc]
            z = self.face_z[foc]
            # x, y, z = self.mesh.geometry.face_centers[foc].T
            nx, ny, nz = normalsS.T
            import inspect
            # print(f"{len(inspect.signature(f[color]).parameters)=}")
            # print(f"{str(inspect.signature(f[color]).parameters)=}")
            # if 'nx' in str(inspect.signature(f[color])):
            if len(inspect.signature(f[color]).parameters) >= 6:
                # ff = f[color](x, y, z, nx[None,:], ny[None,:], nz[None,:])
                ff = f[color](x, y, z, nx, ny, nz)
            else:
                ff = np.vectorize(f[color])(x, y, z)
            b[foc] = ff
        return b
    # matrices
    def masslocal(self):
        dim = self.mesh.dimension
        scalemass = (2 - dim) / (dim + 1) / (dim + 2)
        massloc = np.tile(scalemass, (self.nloc, self.nloc))
        scale = (2 - dim + dim * dim) / (dim + 1) / (dim + 2)
        massloc.reshape((self.nloc * self.nloc))[::self.nloc + 1] = scale
        return massloc
    def _computeMassMatrix(self, coeff=1):
        dim, dV = self.mesh.dimension, self.mesh.geometry.cell_volumes
        return np.einsum('n,kl->nkl', coeff*dV, self.masslocal())
    def computeMassMatrix(self, coeff=1, lumped=False):
        if lumped:
            dim, dV = self.mesh.dimension, self.mesh.geometry.cell_volumes
            nfaces, faces_of_cells = self.mesh.nfaces, self.mesh.topology.faces_of_cells
            mass = coeff/(dim+1)*dV.repeat(dim+1)
            rows = self.mesh.topology.faces_of_cells.ravel()
            return sparse.coo_matrix((mass, (rows, rows)), shape=(nfaces, nfaces)).tocsr()
        nfaces = self.mesh.nfaces
        mass = self._computeMassMatrix(coeff)
        return sparse.coo_matrix((mass.ravel(), (self.rows, self.cols)), shape=(nfaces, nfaces)).tocsr()
    def computeBdryMassMatrix(self, colors=None, coeff=1, lumped=False):
        nfaces, dim = self.mesh.nfaces, self.mesh.dimension
        massloc = barycentric.crbdryothers(dim)
        # lumped = False
        if colors is None: colors = self.mesh.labels.boundary.keys()
        if not isinstance(coeff, dict):
            faces = self.mesh.bdryFaces(colors)
            normalsS = self.mesh.geometry.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            if isinstance(coeff, (int,float)): dS *= coeff
            elif coeff.shape[0]==self.mesh.nfaces: dS *= coeff[faces]
            elif coeff.shape[0]==dS.shape[0]: dS *= coeff
            else: raise  ValueError(f"cannot handle {coeff=}")
            AD = sparse.coo_matrix((dS, (faces, faces)), shape=(nfaces, nfaces))
            if lumped: return AD
            ci = self.mesh.topology.cells_of_faces[faces][:,0]
            foc = self.mesh.topology.faces_of_cells[ci]
            mask = foc != faces[:, np.newaxis]
            # print(f"{mask=}")
            fi = foc[mask].reshape(foc.shape[0], foc.shape[1] - 1)
            # print(f"{massloc=}")
            cols = np.tile(fi, dim).ravel()
            rows = np.repeat(fi, dim).ravel()
            mat = np.einsum('n,kl->nkl', dS, massloc).ravel()
            return AD + sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces))
        assert(isinstance(coeff, dict))
        rows = np.empty(shape=(0), dtype=int)
        cols = np.empty(shape=(0), dtype=int)
        mat = np.empty(shape=(0), dtype=float)
        for color in colors:
            faces = self.mesh.labels.boundary[color]
            normalsS = self.mesh.geometry.normals[faces]
            dS = linalg.norm(normalsS, axis=1)*coeff[color]
            cols = np.append(cols, faces)
            rows = np.append(rows, faces)
            mat = np.append(mat, dS)
            if not lumped:
                ci = self.mesh.topology.cells_of_faces[faces][:,0]
                foc = self.mesh.topology.faces_of_cells[ci]
                mask = foc != faces[:, np.newaxis]
                fi = foc[mask].reshape(foc.shape[0], foc.shape[1] - 1)
                # print(f"{massloc=}")
                cols = np.append(cols, np.tile(fi, dim).ravel())
                rows = np.append(rows, np.repeat(fi, dim).ravel())
                mat = np.append(mat, np.einsum('n,kl->nkl', dS, massloc).ravel())
        # print(f"{mat=}")
        return sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces)).tocsr()

    def massDotBoundary(self, b=None, f=None, colors=None, coeff=1, lumped=False):
        if colors is None:
            colors = self.mesh.labels.boundary.keys()

        if f is None:
            f = np.ones(self.mesh.nfaces)
        elif np.isscalar(f):
            f = np.full(self.mesh.nfaces, f, dtype=float)
        else:
            f = np.asarray(f)
            if f.shape[0] != self.mesh.nfaces:
                raise ValueError(
                    f"CR1 boundary data f must be scalar or have length nfaces: "
                    f"{f.shape=}, {self.mesh.nfaces=}"
                )

        massloc = barycentric.crbdryothers(self.mesh.dimension)

        if not isinstance(coeff, dict):
            faces = self.mesh.bdryFaces(colors)
            normalsS = self.mesh.geometry.normals[faces]
            dS = linalg.norm(normalsS, axis=1)

            if isinstance(coeff, (int, float)):
                dS *= coeff
            elif coeff.shape[0] == self.mesh.nfaces:
                dS *= coeff[faces]
            else:
                dS *= coeff

            if b is None:
                bsum = np.sum(dS * f[faces])
            else:
                b[faces] += dS * f[faces]

            if lumped:
                return bsum if b is None else b

            ci = self.mesh.topology.cells_of_faces[faces][:, 0]
            foc = self.mesh.topology.faces_of_cells[ci]
            mask = foc != faces[:, np.newaxis]
            fi = foc[mask].reshape(foc.shape[0], foc.shape[1] - 1)

            r = np.einsum("n,kl,nl->nk", dS, massloc, f[fi])

            if b is None:
                return bsum + np.sum(r)

            np.add.at(b, fi, r)
            return b
    def computeMatrixJump(self, betart, mode='primal', monotone=False):
        dim, dV, nfaces, ndofs = self.mesh.dimension, self.mesh.geometry.cell_volumes, self.mesh.nfaces, self.nunknowns()
        nloc, dofspercell = self.nlocal(), self.dofspercell()

        cells_of_faces = self.mesh.topology.cells_of_faces
        innerfaces = np.flatnonzero(cells_of_faces[:, 1] >= 0)

        ci0 = cells_of_faces[innerfaces, 0]
        ci1 = cells_of_faces[innerfaces, 1]

        normalsS = self.mesh.geometry.normals[innerfaces]
        dS = linalg.norm(normalsS, axis=1)

        fi0, fi1 = faces_of_cells_not_on_faces(
            self.mesh.topology.faces_of_cells,
            ci0,
            ci1,
            innerfaces,
        )
        A = sparse.coo_matrix((ndofs, ndofs))
        rows0 = np.repeat(fi0, nloc-1).ravel()
        cols0 = np.tile(fi0,nloc-1).ravel()
        rows1 = np.repeat(fi1, nloc-1).ravel()
        cols1 = np.tile(fi1,nloc-1).ravel()

        massloc = barycentric.crbdryothers(self.mesh.dimension)
        if mode == 'primal':
            mat = np.einsum('n,kl->nkl', np.minimum(betart[innerfaces], 0) * dS, massloc).ravel()
            A -= sparse.coo_matrix((mat, (rows0, cols0)), shape=(ndofs, ndofs))
            A += sparse.coo_matrix((mat, (rows0, cols1)), shape=(ndofs, ndofs))
            mat = np.einsum('n,kl->nkl', np.maximum(betart[innerfaces], 0)*dS, massloc).ravel()
            A -= sparse.coo_matrix((mat, (rows1, cols0)), shape=(ndofs, ndofs))
            A += sparse.coo_matrix((mat, (rows1, cols1)), shape=(ndofs, ndofs))
        elif mode =='dual':
            mat = np.einsum('n,kl->nkl', np.minimum(betart[innerfaces], 0) * dS, massloc).ravel()
            A += sparse.coo_matrix((mat, (rows0, cols1)), shape=(ndofs, ndofs))
            A -= sparse.coo_matrix((mat, (rows1, cols1)), shape=(ndofs, ndofs))
            mat = np.einsum('n,kl->nkl', np.maximum(betart[innerfaces], 0) * dS, massloc).ravel()
            A += sparse.coo_matrix((mat, (rows0, cols0)), shape=(ndofs, ndofs))
            A -= sparse.coo_matrix((mat, (rows1, cols0)), shape=(ndofs, ndofs))
        elif mode =='centered':
            mat = np.einsum('n,kl->nkl', betart[innerfaces] * dS, massloc).ravel()
            A += sparse.coo_matrix((mat, (rows0, cols0)), shape=(ndofs, ndofs))
            A -= sparse.coo_matrix((mat, (rows1, cols1)), shape=(ndofs, ndofs))
        else:
            raise ValueError(f"unknown {mode=}")
        return A
    def computeFormJump(self, du, u, betart, mode='primal'):
        cells_of_faces = self.mesh.topology.cells_of_faces
        innerfaces = np.flatnonzero(cells_of_faces[:, 1] >= 0)

        ci0 = cells_of_faces[innerfaces, 0]
        ci1 = cells_of_faces[innerfaces, 1]

        normalsS = self.mesh.geometry.normals[innerfaces]
        dS = linalg.norm(normalsS, axis=1)

        fi0, fi1 = faces_of_cells_not_on_faces(
            self.mesh.topology.faces_of_cells,
            ci0,
            ci1,
            innerfaces,
        )
        massloc = barycentric.crbdryothers(self.mesh.dimension)
        if mode == 'primal':
            mat = np.einsum('n,kl,nl->nk', np.minimum(betart[innerfaces], 0) * dS, massloc, u[fi1]-u[fi0])
            np.add.at(du, fi0, mat)
            mat = np.einsum('n,kl,nl->nk', np.maximum(betart[innerfaces], 0)*dS, massloc, u[fi1]-u[fi0])
            np.add.at(du, fi1, mat)
        elif mode =='dual':
            assert 0
        elif mode =='centered':
            assert 0
        else:
            raise ValueError(f"unknown {mode=}")
    def computeMassMatrixSupg(self, xd, coeff=1):
        raise NotImplemented(f"computeMassMatrixSupg")
    def massDotCell(self, b, f, coeff=1):
        assert f.shape[0] == self.mesh.ncells
        dimension, faces_of_cells, dV = self.mesh.dimension, self.mesh.topology.faces_of_cells, self.mesh.geometry.cell_volumes
        massloc = 1/(dimension+1)
        np.add.at(b, faces_of_cells, (massloc*coeff*dV*f)[:, np.newaxis])
        return b
    def massDot(self, b, f, coeff=1):
        dim, faces_of_cells, dV = self.mesh.dimension, self.mesh.topology.faces_of_cells, self.mesh.geometry.cell_volumes
        scalemass = (2-dim) / (dim+1) / (dim+2)
        massloc = np.tile(scalemass, (self.nloc,self.nloc))
        massloc.reshape((self.nloc*self.nloc))[::self.nloc+1] = (2-dim + dim*dim) / (dim+1) / (dim+2)
        r = np.einsum('n,kl,nl->nk', coeff*dV, massloc, f[faces_of_cells])
        np.add.at(b, faces_of_cells, r)
        return b
    # rhs
    def computeRhsCell(self, b, rhscell):
        if rhscell is None: return b
        if isinstance(rhscell,dict):
            assert set(rhscell.keys())==set(self.mesh.labels.cell.keys())
            dimension, faces_of_cells, dV = self.mesh.dimension, self.mesh.topology.faces_of_cells, self.mesh.geometry.cell_volumes
            scale = 1 / (dimension + 1)
            return b
            scale = 1 / (self.mesh.dimension + 1)
            for label, fct in rhscell.items():
                if fct is None: continue
                cells = self.mesh.labels.cell[label]
                xc, yc, zc = self.mesh.geometry.cell_centers[cells].T
                bC = scale * fct(xc, yc, zc) * dV[cells]
                np.add.at(b, faces_of_cells, bC)
        else:
            fp1 = self.interpolateCell(rhscell)
            self.massDotCell(b, fp1, coeff=1)
        return b
    # postprocess
    def computeErrorL2Cell(self, solexact, uh):
        xc, yc, zc = self.cell_x, self.cell_y, self.cell_z
        ec = solexact(xc, yc, zc) - np.mean(uh[self.mesh.topology.faces_of_cells], axis=1)
        return np.sqrt(np.sum(ec**2* self.mesh.geometry.cell_volumes)), ec
    def computeErrorL2(self, solexact, uh):
        x, y, z = self._xyz_from_points(self.mesh.geometry.face_centers)
        en = solexact(x, y, z) - uh
        Men = np.zeros_like(en)
        return np.sqrt( np.dot(en, self.massDot(Men,en)) ), en
    def computeErrorFluxL2(self, solexact, uh, diffcell=None):
        xc, yc, zc = self.cell_x, self.cell_y, self.cell_z
        graduh = np.einsum('nij,ni->nj', self.cellgrads, uh[self.mesh.topology.faces_of_cells])
        errv = 0
        for i in range(self.mesh.dimension):
            solxi = solexact.d(i, xc, yc, zc)
            if diffcell is None: errv += np.sum((solxi - graduh[:, i]) ** 2 * self.mesh.geometry.cell_volumes)
            else: errv += np.sum(diffcell * (solxi - graduh[:, i]) ** 2 * self.mesh.geometry.cell_volumes)
        return np.sqrt(errv)
    def computeBdryMean(self, u, colors):
        mean, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.labels.boundary[color]
            normalsS = self.mesh.geometry.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            mean[i] = np.sum(dS*u[faces])
        return mean/omega
    def comuteFluxOnRobin(self, u, faces, dS, uR, cR):
        uhmean =  np.sum(dS * u[faces])
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

    def build_scalar_prolongation_to_refined_mesh(self, info):
        return mesh_transfer.cr1_prolongation(info)

    def computeEstimator(self, u, rhs_cell=None, diffcell=None):
        """
        CR1 residual estimator for scalar diffusion:
            eta_K^2 = h_K^2 |K| f_K^2
                    + sum_{F subset dK interior} 0.5 * h_F * |F| * [k grad u . n]^2

        Returns
        -------
        eta : float
        eta2 : (ncells,) ndarray
            squared local indicators
        """
        import numpy as np

        mesh = self.mesh
        points = mesh.geometry.points[:, :2]
        cells = mesh.topology.cells
        faces = mesh.topology.faces
        faces_of_cells = mesh.topology.faces_of_cells
        cells_of_faces = mesh.topology.cells_of_faces

        u = np.asarray(u, dtype=float).reshape(-1)

        ncells = cells.shape[0]
        eta2 = np.zeros(ncells, dtype=float)

        if rhs_cell is None:
            rhs_cell = np.zeros(ncells, dtype=float)
        else:
            rhs_cell = np.asarray(rhs_cell, dtype=float).reshape(-1)

        if diffcell is None:
            diffcell = np.ones(ncells, dtype=float)
        else:
            diffcell = np.asarray(diffcell)

        # ---- cell gradients of CR function
        gradu = np.zeros((ncells, 2), dtype=float)

        for k, tri in enumerate(cells):
            p = points[tri]

            B = np.column_stack((p[1] - p[0], p[2] - p[0]))
            invB = np.linalg.inv(B)

            grad_lam = np.empty((3, 2), dtype=float)
            grad_lam[1] = invB[0]
            grad_lam[2] = invB[1]
            grad_lam[0] = -grad_lam[1] - grad_lam[2]

            # CR basis attached to edge opposite vertex i:
            # phi_i = 1 - 2 lambda_i, hence grad phi_i = -2 grad lambda_i
            for iloc in range(3):
                other = [j for j in range(3) if j != iloc]
                a = int(tri[other[0]])
                b = int(tri[other[1]])
                e = (a, b) if a < b else (b, a)

                gf = None
                for cand in faces_of_cells[k]:
                    fa, fb = map(int, faces[cand])
                    ec = (fa, fb) if fa < fb else (fb, fa)
                    if ec == e:
                        gf = int(cand)
                        break

                if gf is None:
                    raise RuntimeError("Could not match local CR face dof to opposite edge")

                gradu[k] += u[gf] * (-2.0 * grad_lam[iloc])

        # ---- cell residual term: P1/CR1 has zero Laplacian cellwise
        vols = mesh.geometry.cell_volumes
        for k, tri in enumerate(cells):
            p = points[tri]
            hK = max(
                np.linalg.norm(p[1] - p[0]),
                np.linalg.norm(p[2] - p[1]),
                np.linalg.norm(p[0] - p[2]),
            )
            eta2[k] += hK ** 2 * vols[k] * rhs_cell[k] ** 2

        # ---- interior flux jumps
        for f, adj in enumerate(cells_of_faces):
            k0, k1 = map(int, adj)
            if k0 < 0 or k1 < 0:
                continue

            a, b = map(int, faces[f])
            xa, xb = points[a], points[b]
            t = xb - xa
            hF = np.linalg.norm(t)
            if hF == 0.0:
                continue

            n = np.array([t[1], -t[0]]) / hF

            q0 = diffcell[k0] * gradu[k0]
            q1 = diffcell[k1] * gradu[k1]

            jump = np.dot(q0 - q1, n)

            contrib = 0.5 * hF * hF * jump ** 2
            eta2[k0] += 0.5 * contrib
            eta2[k1] += 0.5 * contrib

        eta = float(np.sqrt(np.sum(eta2)))
        return eta, eta2

    def to_p1(self, uface):
        npoints = self.mesh.geometry.points.shape[0]

        vals = np.zeros(npoints)
        cnt = np.zeros(npoints)

        for f, (a, b) in enumerate(self.mesh.topology.faces):
            vals[a] += uface[f]
            vals[b] += uface[f]
            cnt[a] += 1
            cnt[b] += 1

        return vals / np.maximum(cnt, 1)
# ------------------------------------- #
if __name__ == '__main__':
    from FEM2D.meshes_new import testmeshes
    from FEM2D.meshes_new import plotmesh
    import matplotlib.pyplot as plt
    mesh = testmeshes.backwardfacingstep(h=0.2)
    fem = CR1(mesh=mesh)
    u = fem.test()
    plotmesh.meshWithBoundaries(mesh)
    plotmesh.meshWithData(mesh, point_data={'u':u}, title="CR1 Test", alpha=1)
    plt.show()
