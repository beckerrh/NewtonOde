import numpy as np

def eliminate_matrix_symmetric(A, dofs, diag=1.0):
    dofs = np.asarray(dofs, dtype=int)

    A = A.tolil()
    A[dofs, :] = 0.0
    A[:, dofs] = 0.0
    A[dofs, dofs] = diag
    return A.tocsr()


def eliminate_residual(r, dofs, value=0.0):
    r = r.copy()
    r[np.asarray(dofs, dtype=int)] = value
    return r