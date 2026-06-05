import numpy as np
import scipy, scipy.special
import itertools as it

# ------------------------------------- #
def coords(points, simplex):
    points = np.asarray(points)
    simplex = np.asarray(simplex)

    if points.ndim == 1:
        points = points[None, :]

    nsimplex_vertices = simplex.shape[0]
    dim = nsimplex_vertices - 1

    simplex = simplex[:, :dim]
    points = points[:, :dim]

    A = np.vstack((simplex.T, np.ones(nsimplex_vertices)))
    B = np.vstack((points.T, np.ones(points.shape[0])))

    return np.linalg.solve(A, B).T
# ------------------------------------- #
def tensor(d, k):
    A = np.ones(shape=k*[d+1])
    facd = np.prod(np.arange(d + 1, d + k + 1))
    # print(f"{np.arange(d + 1, d + k + 1)=} {facd=}")
    for i in it.product(np.arange(d+1), repeat=k):
        A[i] = np.prod(scipy.special.factorial(np.bincount(i)))/facd
    return A

# ------------------------------------- #
def crbdryothers(d):
    massloc = -np.ones(shape=(d, d)) / (d + 1)
    massloc[np.diag_indices(d)] = (d - 1) / (d + 1)
    return massloc


# ------------------------------------- #
if __name__ == '__main__':
    print(f"{tensor(d=3, k=2)=}")
    print(f"{tensor(d=3, k=1)=}")