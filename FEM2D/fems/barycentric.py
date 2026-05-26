import numpy as np
import scipy, scipy.special
import itertools as it

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