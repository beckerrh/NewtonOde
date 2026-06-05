import numpy as np

def normalize_reaction(reaction, ncells, ncomp):
    r = np.asarray(reaction)

    if r.ndim == 0:
        return "scalar", np.full(ncells, float(r))

    if r.shape == (ncells,):
        return "scalar", r

    if r.shape == (ncomp, ncells):
        return "diagonal", r

    if r.shape == (ncomp,):
        return "diagonal", np.repeat(r[:, None], ncells, axis=1)

    if r.shape == (ncomp, ncomp):
        return "coupled", np.repeat(r[:, :, None], ncells, axis=2)

    if r.shape == (ncomp, ncomp, ncells):
        return "coupled", r

    raise ValueError(f"invalid reaction shape {r.shape=}")