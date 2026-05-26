# -*- coding: utf-8 -*-
"""Small edge/topology helpers used by NVB refinement."""

import numpy as np


LOCAL_EDGES = ((0, 1), (1, 2), (2, 0))


def sorted_edge(a, b):
    return (a, b) if a < b else (b, a)

# def sorted_edge(i, j):
#     i = int(i)
#     j = int(j)
#     if i == j:
#         raise ValueError(f"Degenerate edge ({i}, {j})")
#     return (i, j) if i < j else (j, i)


def compute_edges(cells):
    """
    Compute unique unoriented edges and element-to-edge connectivity.

    Parameters
    ----------
    cells : ndarray, shape (ncells, 3)

    Returns
    -------
    edges : ndarray, shape (nfaces, 2)
        Unique sorted vertex pairs.
    element2edges : ndarray, shape (ncells, 3)
        Local edge ids for edges (0,1), (1,2), (2,0).
    """
    edge_dict = {}
    edges = []
    element2edges = np.zeros((cells.shape[0], 3), dtype=int)

    for icell, tri in enumerate(cells):
        for iloc, (i, j) in enumerate(LOCAL_EDGES):
            edge = sorted_edge(tri[i], tri[j])
            if edge not in edge_dict:
                edge_dict[edge] = len(edges)
                edges.append(edge)
            element2edges[icell, iloc] = edge_dict[edge]

    return np.asarray(edges, dtype=int), element2edges


def face_dict(faces):
    """Map sorted face vertex-pairs to face indices."""
    return {sorted_edge(i, j): k for k, (i, j) in enumerate(faces)}
