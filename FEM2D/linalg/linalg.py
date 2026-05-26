import numpy as np
import scipy.sparse.linalg as splinalg
import scipy.sparse as sparse
import time
from FEM2D.linalg import iterationcounter

scipysolvers=['scipy_gmres','scipy_lgmres','scipy_gcrotmk','scipy_bicgstab','scipy_cgs', 'scipy_cg']
pyamgsolvers=['pyamg_fgmres','pyamg_bicgstab', 'pyamg_cg']
# pyamg_gmres seems to not work correctly
strangesolvers=['gmres']
othersolvers=['idr']


#=================================================================#
class MassMatrixIncompressible:
    def __init__(self, app, M):
        self.app = app
        self.M = M
    def addToStokes(self, d, A):
        if self.app.singleA:
            A += d * self.M
        else:
            A += matrix2systemdiagonal(d * self.M, self.app.ncomp, self.app.stack_storage)
        return A
    def dot(self, du, d, u):
        # print(f"{d=}")
        for i in range(self.app.ncomp):
            self.app.vectorview.get(0,i,du)[:] += d * self.M @ self.app.vectorview.get(0,i,u)
        return du
        n = self.M.shape[0]
        v, p = self.app._split(u)
        dv, dp = self.app._split(du)
        # print(f"{n=} {dv.shape=} {np.linalg.norm(u)=} {np.linalg.norm(self.M.data)=}")
        if self.app.stack_storage:
            n = self.M.shape[0]
            for i in range(self.app.ncomp):
                dv[i*n:(i+1)*n] += d * self.M.dot(v[i*n:(i+1)*n])
        else:
            for i in range(self.app.ncomp):
                dv[i::self.app.ncomp] += d * self.M.dot(v[i::self.app.ncomp])
        # print(f"{np.linalg.norm(du)=} {np.linalg.norm(u)=}")
        return du


#=================================================================#
def matrix2systemdiagonal(A, ncomp, stack_storage):
    """
    creates a blockmatrix with A on the diaganals
    hyothesis: vector is stored as v[::icomp] and NOT one after the other
    :param self:
    :param A:
    :param ncomp:
    :return:
    """
    if stack_storage:
        n = A.shape[0]
        # N = sparse.coo_matrix((n,n))
        # N = sparse.dok_matrix((n,n))
        N = sparse.csr_matrix((n,n))
        if ncomp==2:
            return sparse.hstack([sparse.vstack([A,N]), sparse.vstack([N,A])])
        return sparse.hstack([sparse.vstack([A, N, N]), sparse.vstack([N, A, N]), sparse.vstack([N, N, A])])
    A = A.tocoo()
    data, row, col = A.data, A.row, A.col
    n = A.shape[0]
    assert n == A.shape[1]
    data2 = np.repeat(data, ncomp)
    nr = row.shape[0]
    row2 = np.repeat(ncomp * row, ncomp) + np.tile(np.arange(ncomp), nr).ravel()
    col2 = np.repeat(ncomp * col, ncomp) + np.tile(np.arange(ncomp), nr).ravel()
    return sparse.coo_matrix((data2, (row2, col2)), shape=(ncomp * n, ncomp * n)).tocsr()
def diagmatrix2systemdiagmatrix(A, ncomp, stack_storage):
    """
    creates a blockmatrix with A on the diaganals
    hyothesis: vector is stored as v[::icomp] and NOT one after the other
    :param self:
    :param A:
    :param ncomp:
    :return:
    """
    data = A.data
    n = A.shape[0]
    if stack_storage:
        data = np.tile(data, ncomp)
    else:
        data = np.repeat(data, ncomp)
    return sparse.diags(data, offsets=(0), shape=(ncomp*n, ncomp*n))

#=================================================================#
class DiagonalScaleSolver():
    def __repr__(self):
        return f"{self.__class__.__name__}"
    def __init__(self, coeff):
        n = len(coeff)
        self.BP = sparse.diags(coeff, offsets=(0), shape=(n,n))
    def solve(self, b):
        return self.BP.dot(b)
#-------------------------------------------------------------------#
def getLinearSolver(**kwargs):
    # print(f"{kwargs=}")
    method = kwargs.pop('method', 'pyamg')
    matrix = kwargs.pop('matrix', None)
    if method in scipysolvers or method in pyamgsolvers or method in othersolvers:
        if matrix is not None:
            return ScipySolve(matrix=matrix, method=method, **kwargs)
        else:
            return ScipySolve(method=method, **kwargs)
    elif method == "spsolve":
        return ScipySpSolve(matrix=matrix)
    elif method == "pyamg":
        return Pyamg(matrix, **kwargs)
    else:
        raise ValueError(f"unknwown {method=} not in ['spsolve', 'pyamg', {pyamgsolvers}] nor {scipysolvers=}")
    if len(kwargs.keys()):
        raise ValueError(f"*** unused arguments {kwargs=}")

#-------------------------------------------------------------------#
def getLinearSolversAndTest(**kwargs):
    """
    :param kwargs: if args is dict build the correspong solver
    otherwise if args is list, choose the best solver in the list
    :return:
    """
    args = kwargs.pop('args')
    if isinstance(args, dict):
        if len(kwargs): raise ValueError(f"*** unused keys {kwargs}")
        return getLinearSolver(args)
    assert isinstance(args, list)
    maxiter = args.pop('maxiter', 50)
    verbose = args.pop('verbose', 0)
    reduction = args.pop('reduction', 0.01)
    rtol = args.pop('rtol') if 'rtol' in args else 0.1*reduction
    solvers = {}
    for arg in args:
        solvers[arg] = getLinearSolver(arg)
        n = solvers[arg].shape[0]
    b = np.random.random(n)
    b /= np.linalg.norm(b)
    analysis = {}
    for solvername, solver in solvers.items():
        t0 = time.time()
        res = solver.testsolve(b=b, maxiter=maxiter, rtol=rtol)
        t = time.time() - t0
        monotone = np.all(np.diff(res) < 0)
        if len(res)==1:
            if res[0] > rtol:
                print(f"no convergence in {solvername=} {res=}")
                continue
            iterused = 1
        else:
            rho = np.power(res[-1]/res[0], 1/len(res))
            if not monotone:
                print(f"***VelcoitySolver {solvername} not monotone {rho=}")
                continue
            if rho > 0.8:
                print(f"***VelcoitySolver {solvername} bad {rho=}")
                continue
            iterused = int(np.log(reduction)/np.log(rho))+1
        treq = t/len(res)*iterused
        analysis[solvername] = (iterused, treq)
    # print(f"{self.analysis=}")
    if verbose:
        for solvername, val in analysis.items():
            print(f"{solvername=} {val=}")
    if len(analysis)==0: raise ValueError('*** no working solver found')
    ibest = np.argmin([v[1] for v in analysis.values()])
    solverbest = list(analysis.keys())[ibest]
    if verbose:
        print(f"{solverbest=}")
    return solvers[solverbest], analysis[solverbest][0]
#=================================================================#
class ScipySpSolve():
    def __init__(self, **kwargs):
        self.matrix = kwargs.pop('matrix', None)
        self.niter = 1
        self.maxiter = -1
    def solve(self, A=None, b=None, maxiter=None, rtol=None, atol=None, x0=None, verbose=None):
        if A is None: A=self.matrix
        if hasattr(A, 'to_single_matrix'):
            A = A.to_single_matrix()
        return splinalg.spsolve(A, b)
    def update(self, A):
        self.matrix = A
    def testsolve(self, b, maxiter, rtol):
        splinalg.spsolve(self.matrix, b)
        return [0]
#=================================================================#
class IterativeSolver():
    def __repr__(self):
        return f"{self.method}_{self.maxiter}_{self.rtol}"
    def __init__(self, kwargs):
        self.args = {}
        # print(f"# {kwargs}")
        self.scale = kwargs.pop('scale', False)
        self.atol = kwargs.pop('atol', 1e-14)
        self.rtol = kwargs.pop('rtol', 1e-8)
        self.maxiter = kwargs.pop('maxiter', 100)
        disp = kwargs.pop('disp', 0)
        self.counter = iterationcounter.IterationCounter(name=kwargs.pop('counter','')+str(self), disp=disp)
        self.args['callback'] = self.counter
    def solve(self, A=None, b=None, maxiter=None, rtol=None, atol=None, x0=None, verbose=None):
        if maxiter is None: maxiter = self.maxiter
        if rtol is None: rtol = self.rtol
        if hasattr(self, 'counter'):
            self.counter.reset()
            self.args['callback'] = self.counter
        if A is not None:
            if self.scale and hasattr(A,'scale_matrix'):
                A.scale_vec(b)
        self.args['b'] = b
        self.args['maxiter'] = maxiter
        self.args['x0'] = x0
        self.args['tol'] = rtol
        res  = self.solver(**self.args)
        if hasattr(self, 'counter'):
            self.niter = self.counter.niter
        else:
            self.niter = -1
        sol = res[0] if isinstance(res, tuple) else res
        if A is not None:
            if self.scale and hasattr(A,'scale_matrix'):
                A.scale_vec(sol)
        return sol

    def testsolve(self, b, maxiter, rtol):
        counter = tools.iterationcounter.IterationCounterWithRes(name=str(self), callback_type='x', disp=0, b=b, A=self.matvec)
        args = self.args.copy()
        args['callback'] = counter
        args['maxiter'] = maxiter
        args['tol'] = rtol
        args['b'] = b
        res = self.solver(**args)
        return counter.history
#=================================================================#
class ScipySolve(IterativeSolver):
    def __init__(self, **kwargs):
        self.method = kwargs.pop('method')
        super().__init__(kwargs)
        # if self.method in strangesolvers: raise ValueError(f"method '{self.method}' is i strange scipy solver")
        self.n = kwargs.pop('n', None)
        if "preconditioner" in kwargs:
            # n = kwargs.pop('n')
            self.preconditioner = kwargs.pop('preconditioner')
            self.M = splinalg.LinearOperator(shape=(self.n, self.n), matvec=self.preconditioner.solve)
        # else:
        # self.M = None
        # if "prec" in kwargs:
        #     self.M = kwargs.pop("prec")
        if "matrix" in kwargs:
            self.matvec = kwargs.pop('matrix')
            if not "matvecprec" in kwargs:
                fill_factor = kwargs.pop("fill_factor", 2)
                drop_tol = kwargs.pop("fill_factor", 0.01)
                spilu = splinalg.spilu(self.matvec.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
                if not hasattr(self, "M"): self.M = splinalg.LinearOperator(self.matvec.shape, lambda x: spilu.solve(x))
        else:
            if self.n is None: raise ValueError(f"need 'n' if no matrix given")
            # n = kwargs.pop('n')
            # raise ValueError(f"@@@@{n=}")
            self.matvec = splinalg.LinearOperator(shape=(self.n, self.n), matvec=kwargs.pop('matvec'))
        # if "preconditioner" in kwargs:
        #     # n = kwargs.pop('n')
        #     if kwargs.pop('preconditioner') == "scipy.sparse.linalg.spilu":
        #         self.preconditioner = sparse.linalg.spilu()
        #         self.M = splinalg.LinearOperator(shape=(self.n, self.n), matvec=self.preconditioner.solve)
        # else:
        #     self.M = None
        # self.args = {"A": self.matvec, "M":self.M, "atol":self.atol}
        self.args['A'] = self.matvec
        self.args['M'] = self.M
        name = self.method
        if self.method=='scipy_gcrotmk':
            self.args['m'] = kwargs.pop('m', 5)
            self.args['truncate'] = kwargs.pop('truncate', 'smallest')
            self.solver = splinalg.gcrotmk
            name += '_' + str(self.args['m'])
        elif self.method=='scipy_lgmres':
            self.args['inner_m'] = kwargs.pop('m', 20)
        if self.method in scipysolvers:
            self.solver = eval('splinalg.'+self.method[6:])
            self.args['atol'] = self.atol
        elif self.method in pyamgsolvers:
            import pyamg
            self.solver = eval('pyamg.krylov.' + self.method[6:])
        elif self.method == 'idr':
            # import scipy.sparse.linalg.isolve
            import idrs
            self.solver = eval('idrs.idrs')
        else:
            raise ValueError("*** unknown {self.method=}")
        if len(kwargs.keys()):
            raise ValueError(f"*** unused arguments {kwargs=}")
    def update(self, A):
        # print(f"update {self.__class__.__name__=}")
        self.matvec = splinalg.LinearOperator(shape=(self.n, self.n), matvec=A.matvec)
        self.preconditioner.update(A)
        self.M = splinalg.LinearOperator(shape=(self.n, self.n), matvec=self.preconditioner.solve)
        self.args['A'] = self.matvec
        self.args['M'] = self.M


#=================================================================#
class Pyamg(IterativeSolver):
    def __repr__(self):
        s = super().__repr__()
        return s + f"pyamg_{self.type}_{self.smoother}_{str(self.accel)}"
    def __init__(self, A, **kwargs):
        try:
            import pyamg
        except:
            raise ImportError(f"*** pyamg not found ***")
        assert A is not None
        self.method = 'pyamg'
        nsmooth = kwargs.pop('nsmooth', 1)
        symmetric = kwargs.pop('symmetric', False)
        self.smoother = kwargs.pop('smoother', 'gauss_seidel')
        # pyamgargs = {'B': pyamg.solver_configuration(A, verb=False)['B']}
        # pyamgargs = kwargs.pop("pyamgargs", {})
        # print(f"{pyamgargs=}")
        pyamgargs = {}
        smoother = (self.smoother, {'sweep': 'symmetric', 'iterations': nsmooth})
        if symmetric:
            smooth = ('energy', {'krylov': 'cg'})
        else:
            smooth = ('energy', {'krylov': 'fgmres'})
            pyamgargs['symmetry'] = 'nonsymmetric'
        pyamgargs['presmoother'] = smoother
        pyamgargs['postsmoother'] = smoother
        pyamgargs['smooth'] = smooth
        pyamgargs['coarse_solver'] = 'splu'
        type = kwargs.pop('pyamgtype', 'aggregation')
        if type == 'aggregation':
            self.mlsolver = pyamg.smoothed_aggregation_solver(A, **pyamgargs)
        elif type == 'rootnode':
            self.mlsolver = pyamg.rootnode_solver(A, **pyamgargs)
        else:
            raise ValueError(f"unknown {type=}")
        self.type = type
        self.pyamgargs = pyamgargs
        self.solver = self.mlsolver.solve
        #        cycle : {'V','W','F','AMLI'}
        if symmetric: accel='cg'
        else: accel = 'fgmres'
        self.accel = kwargs.pop('accel', accel)
        super().__init__(kwargs)
        self.args['cycle'] = kwargs.pop('cycle', 'V')
        self.args['accel'] = self.accel
        if len(kwargs.keys()):
            raise ValueError(f"*** unused arguments {kwargs=}")
    def update(self, A):
        import pyamg
        if self.type == 'aggregation':
            self.mlsolver = pyamg.smoothed_aggregation_solver(A, **self.pyamgargs)
        elif self.type == 'rootnode':
            self.mlsolver = pyamg.rootnode_solver(A, **self.pyamgargs)
        else:
            raise ValueError(f"unknown {type=}")
        self.solver = self.mlsolver.solve
