
def _refine_cell_nvb_iterative_keyed(tri, refedge, marked_keys, points, new_points, key_to_mid, nkey):
    stack = [(int(tri[0]), int(tri[1]), int(tri[2]), int(refedge[0]), int(refedge[1]))]

    children = []
    child_refedges = []

    while stack:
        a, b, c, r0, r1 = stack.pop()

        if r0 < r1:
            key = r0 * nkey + r1
        else:
            key = r1 * nkey + r0

        if key not in marked_keys:
            children.append([a, b, c])
            child_refedges.append((r0, r1))
            continue

        m = key_to_mid[key]

        if a != r0 and a != r1:
            z = a
        elif b != r0 and b != r1:
            z = b
        else:
            z = c

        stack.append((z, m, r1, z, r1))
        stack.append((z, r0, m, z, r0))

    return children, child_refedges
def _refine_cell_nvb_iterative(tri, refedge, marked_edges, points, new_points, edge_to_mid):
    """
    Refine one triangle by iterative NVB until no marked refinement edge remains.

    Parameters
    ----------
    tri : list[int] or tuple[int, int, int]
        Triangle vertices.
    refedge : tuple[int, int]
        Current refinement edge.
    marked_edges : set[tuple[int, int]]
        Globally marked edges, stored sorted.
    points, new_points :
        Unused here, kept for signature compatibility.
    edge_to_mid : dict[tuple[int, int], int]
        Sorted edge -> midpoint vertex.

    Returns
    -------
    children : list[list[int]]
    child_refedges : list[tuple[int, int]]
    """
    stack = [(int(tri[0]), int(tri[1]), int(tri[2]), int(refedge[0]), int(refedge[1]))]

    children = []
    child_refedges = []

    while stack:
        a, b, c, r0, r1 = stack.pop()

        # sorted refinement edge
        if r0 < r1:
            e0, e1 = r0, r1
        else:
            e0, e1 = r1, r0

        edge = (e0, e1)

        if edge not in marked_edges:
            children.append([a, b, c])
            child_refedges.append((r0, r1))
            continue

        m = edge_to_mid[edge]

        # third vertex opposite to refinement edge
        if a != r0 and a != r1:
            z = a
        elif b != r0 and b != r1:
            z = b
        else:
            z = c

        # NVB split of triangle (r0, r1, z) along edge (r0, r1)
        #
        # children:
        #   T0 = (z, r0, m)
        #   T1 = (z, m, r1)
        #
        # new refinement edges:
        #   (z, r0) and (z, r1)
        #
        stack.append((z, m, r1, z, r1))
        stack.append((z, r0, m, z, r0))

    return children, child_refedges
