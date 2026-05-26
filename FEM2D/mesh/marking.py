# -*- coding: utf-8 -*-
import numpy as np


def dorfler_marking(eta, theta=0.5, squared=False):
    """
    Minimal Dörfler marking.

    Parameters
    ----------
    eta : array_like
        Cell indicators eta_K or eta_K^2.
    theta : float
        Marking parameter in (0,1].
    squared : bool
        If False, eta is squared internally.

    Returns
    -------
    marked : bool ndarray
    """
    eta = np.asarray(eta, dtype=float)

    if eta.ndim != 1:
        raise ValueError(f"eta must be one-dimensional, got {eta.shape=}")

    if not 0 < theta <= 1:
        raise ValueError(f"theta must satisfy 0 < theta <= 1, got {theta=}")

    eta2 = eta if squared else eta**2
    total = np.sum(eta2)

    marked = np.zeros(eta2.shape, dtype=bool)

    if total <= 0:
        return marked

    idx = np.argsort(eta2)[::-1]
    cumulative = np.cumsum(eta2[idx])

    nmark = np.searchsorted(cumulative, theta * total) + 1
    marked[idx[:nmark]] = True

    return marked