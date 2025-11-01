import numpy as np

#==================================================================
def mesh(t0, t1, n, type='uniform', alpha=0.01):
    if type == 'uniform':
        return np.linspace(t0, t1, n)
    else:
        w = np.cumsum(np.random.rand(n))
        w = (w - w[0])/(w[-1]-w[0])
        return t0+ (t1-t0) * w

# ==================================================================
def adapt_mesh(mesh, eta, theta=0.75):
    assert len(mesh) == len(eta)+1
    indsort = np.argsort(eta)[::-1]
    estacc = np.add.accumulate(eta[indsort])
    etatotal = estacc[-1]
    index = np.searchsorted(estacc, theta*etatotal)
    nt = len(mesh)
    refine = np.sort(indsort[:index+1])
    non_refined = np.setdiff1d(np.arange(nt - 1), refine)
    # print(f"{refine=}")
    # print(f"{non_refined=}")

    # --- 2. Create new points
    midpoints = 0.5 * (mesh[refine] + mesh[refine + 1])
    meshnew_unsorted = np.concatenate([mesh, midpoints])
    perm = np.argsort(meshnew_unsorted)
    meshnew = meshnew_unsorted[perm]

    # --- 3. Get sorted indices of all original and midpoint points
    invperm = np.empty_like(perm)
    invperm[perm] = np.arange(len(perm))
    # --- find new indices of original points and midpoints
    old_pos = invperm[:nt]  # new positions of old mesh nodes
    mid_pos = invperm[nt:]  # new positions of midpoints (same order as refine)

    # --- build maps
    refined_map = np.empty((len(refine), 3), dtype=int)
    for k, i in enumerate(refine):
        left_new = old_pos[i]
        mid_new = mid_pos[k]
        right_new = old_pos[i + 1]
        refined_map[k] = [i, left_new, mid_new] if left_new < mid_new else [i, mid_new, left_new]
        # this guarantees correct left-right order in the new mesh

    non_refined_map = np.empty((len(non_refined), 2), dtype=int)
    for k, i in enumerate(non_refined):
        left_new = old_pos[i]
        right_new = old_pos[i + 1]
        non_refined_map[k] = [i, min(left_new, right_new)]

    # print(f"{mesh=}\n{meshnew=}")
    # for i, l, m in refined_map:
    #     print(f"Old interval {i}: new left={meshnew[l]:.3f}, new mid={meshnew[m]:.3f}")
    return meshnew, (refined_map, non_refined_map)

    old_pos = invperm[:nt]  # new positions of old mesh points
    mid_pos = invperm[nt:][np.argsort(refine)]  # new positions of midpoints (sorted like refine)

    # --- 4. Build mapping arrays directly
    refined_map = np.stack(
        [refine,
         np.minimum(old_pos[refine], mid_pos),
         np.maximum(old_pos[refine], mid_pos)],
        axis=1
    )
    non_refined_map = np.stack(
        [non_refined,
         np.minimum(old_pos[non_refined], old_pos[non_refined + 1])],
        axis=1
    )
    # print(f"{refined_map=} {non_refined_map=}")
    return meshnew, (refined_map, non_refined_map)


