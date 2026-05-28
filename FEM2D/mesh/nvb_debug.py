from collections import defaultdict, deque


def _sedge(a, b):
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def _cell_edges(tri):
    a, b, c = map(int, tri)
    return {
        _sedge(a, b),
        _sedge(b, c),
        _sedge(c, a),
    }


def check_refedges(mesh, where=""):
    for icell, tri in enumerate(mesh.topology.cells):
        tri = list(map(int, tri))
        e0, e1 = map(int, mesh.refedges[icell])

        cell_edges = {
            _sedge(tri[0], tri[1]),
            _sedge(tri[1], tri[2]),
            _sedge(tri[2], tri[0]),
        }

        if _sedge(e0, e1) not in cell_edges:
            print("BAD REFEDGE", where)
            print("  icell =", icell)
            print("  tri   =", tri)
            print("  ref   =", (e0, e1))
            print("  cell_edges =", cell_edges)
            raise RuntimeError("mesh.refedges is not aligned with mesh.topology.cells")

def debug_nonmanifold_edges(cells):
    edge_to_cells = defaultdict(list)
    for ic, c in enumerate(cells):
        c = list(map(int, c))
        for a, b in [(c[0], c[1]), (c[1], c[2]), (c[2], c[0])]:
            e = (a, b) if a < b else (b, a)
            edge_to_cells[e].append(ic)

    for e, cs in edge_to_cells.items():
        if len(cs) > 2:
            print("NONMANIFOLD EDGE", e, "cells", cs)
            for ic in cs:
                print("  cell", ic, cells[ic])
            raise ValueError(f"Nonmanifold edge {e}: used by {len(cs)} cells")
