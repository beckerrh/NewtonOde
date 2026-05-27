import scipy.sparse

def p1_prolongation(info):
    rows, cols, data = [], [], []

    for i in range(info.old_npoints):
        rows.append(i)
        cols.append(i)
        data.append(1.0)

    for m, (a, b) in info.midpoint_parents.items():
        rows += [m, m]
        cols += [a, b]
        data += [0.5, 0.5]

    return scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(info.new_npoints, info.old_npoints),
    )