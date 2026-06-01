import scipy.sparse


def p1_prolongation(info):
    rows, cols, data = [], [], []

    for i in range(info.old_npoints):
        rows.append(i)
        cols.append(i)
        data.append(1.0)

    for m, (a, b) in info.midpoint_parents.items():
        rows.extend([m, m])
        cols.extend([a, b])
        data.extend([0.5, 0.5])

    return scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(info.new_npoints, info.old_npoints),
    )


def interpolate_p1_to_refined_mesh(info, u_old):
    P = p1_prolongation(info)
    return  P @ u_old
