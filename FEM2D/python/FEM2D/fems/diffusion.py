import numpy as np


def normalize_diffusion(diff, ncells, ncomp=1, dim=2):
    """
    Returns canonical diffusion with shape:

        scalar:     (ncells,)
        diagonal:   (ncells, ncomp)
        full:       (ncells, ncomp, ncomp)

    This is component diffusion, still scalar in physical space.
    """
    if diff is None:
        return "scalar", np.ones(ncells)

    diff = np.asarray(diff)

    if diff.ndim == 0:
        return "scalar", np.full(ncells, float(diff))

    if diff.shape == (ncells,):
        return "scalar", diff

    if diff.shape == (ncomp,):
        return "diagonal", np.tile(diff[None, :], (ncells, 1))

    if diff.shape == (ncells, ncomp):
        return "diagonal", diff

    if diff.shape == (ncomp, ncomp):
        return "matrix", np.tile(diff[None, :, :], (ncells, 1, 1))

    if diff.shape == (ncells, ncomp, ncomp):
        return "matrix", diff

    raise ValueError(f"Bad diffusion shape {diff.shape=}, {ncells=}, {ncomp=}")

import numpy as np


def normalize_diffusion(diff, ncells, ncomp=1, dim=2):
    """
    Returns canonical diffusion with shape:

        scalar:     (ncells,)
        diagonal:   (ncells, ncomp)
        full:       (ncells, ncomp, ncomp)

    This is component diffusion, still scalar in physical space.
    """
    if diff is None:
        return "scalar", np.ones(ncells)

    diff = np.asarray(diff)

    if diff.ndim == 0:
        return "scalar", np.full(ncells, float(diff))

    if diff.shape == (ncells,):
        return "scalar", diff

    if diff.shape == (ncomp,):
        return "diagonal", np.tile(diff[None, :], (ncells, 1))

    if diff.shape == (ncells, ncomp):
        return "diagonal", diff

    if diff.shape == (ncomp, ncomp):
        return "matrix", np.tile(diff[None, :, :], (ncells, 1, 1))

    if diff.shape == (ncells, ncomp, ncomp):
        return "matrix", diff

    raise ValueError(f"Bad diffusion shape {diff.shape=}, {ncells=}, {ncomp=}")