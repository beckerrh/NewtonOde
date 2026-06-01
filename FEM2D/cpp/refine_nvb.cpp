#include "refine_nvb.hpp"

#include <array>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

using i64 = std::int64_t;

static inline i64 edge_key(i64 a, i64 b, i64 nkey)
{
    if (a > b) std::swap(a, b);
    return a * nkey + b;
}

/*-----------------------------------------------------------*/
struct Edge {
    int a;
    int b;

    Edge(int i, int j) {
        if (i < j) { a = i; b = j; }
        else       { a = j; b = i; }
    }

    bool operator==(const Edge& other) const {
        return a == other.a && b == other.b;
    }
};

struct EdgeHash {
    std::size_t operator()(const Edge& e) const {
        return (static_cast<std::size_t>(e.a) << 32)
             ^ static_cast<std::size_t>(e.b);
    }
};

/*-----------------------------------------------------------*/
py::dict unique_edges(py::array_t<int> edges)
{
    auto E = edges.unchecked<2>();

    std::unordered_map<Edge, int, EdgeHash> edge_to_id;
    std::vector<std::array<int, 2>> unique;

    for (py::ssize_t i = 0; i < E.shape(0); ++i) {
        Edge e(E(i, 0), E(i, 1));

        if (edge_to_id.find(e) == edge_to_id.end()) {
            int id = static_cast<int>(unique.size());
            edge_to_id[e] = id;
            unique.push_back({e.a, e.b});
        }
    }

    py::array_t<int> out({
        static_cast<py::ssize_t>(unique.size()),
        static_cast<py::ssize_t>(2)
    });

    auto O = out.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(unique.size()); ++i) {
        O(i, 0) = unique[i][0];
        O(i, 1) = unique[i][1];
    }

    py::dict result;
    result["edges"] = out;
    result["n_edges"] = static_cast<int>(unique.size());
    return result;
}

/*-----------------------------------------------------------*/
py::dict build_midpoints(
    py::array_t<double> points,
    py::array_t<int> edges
)
{
    auto P = points.unchecked<2>();
    auto E = edges.unchecked<2>();

    const py::ssize_t n_points = P.shape(0);
    const py::ssize_t dim = P.shape(1);
    const py::ssize_t n_edges = E.shape(0);

    py::array_t<double> new_points({
        n_points + n_edges,
        dim
    });

    auto NP = new_points.mutable_unchecked<2>();

    // copy old points
    for (py::ssize_t i = 0; i < n_points; ++i) {
        for (py::ssize_t d = 0; d < dim; ++d) {
            NP(i, d) = P(i, d);
        }
    }

    // midpoint ids
    py::array_t<int> midpoint_ids(n_edges);
    auto MID = midpoint_ids.mutable_unchecked<1>();

    for (py::ssize_t k = 0; k < n_edges; ++k) {

        int a = E(k, 0);
        int b = E(k, 1);

        int mid_id = static_cast<int>(n_points + k);
        MID(k) = mid_id;

        for (py::ssize_t d = 0; d < dim; ++d) {
            NP(mid_id, d) = 0.5 * (P(a, d) + P(b, d));
        }
    }

    py::dict result;
    result["points"] = new_points;
    result["midpoint_ids"] = midpoint_ids;

    return result;
}


/*-----------------------------------------------------------*/
static inline bool has_key(
    const std::unordered_map<i64, i64>& edge_to_mid,
    i64 a,
    i64 b,
    i64 nkey
)
{
    return edge_to_mid.find(edge_key(a, b, nkey)) != edge_to_mid.end();
}


static inline i64 mid_of(
    const std::unordered_map<i64, i64>& edge_to_mid,
    i64 a,
    i64 b,
    i64 nkey
)
{
    auto it = edge_to_mid.find(edge_key(a, b, nkey));
    if (it == edge_to_mid.end()) {
        throw std::runtime_error("missing midpoint for refined edge");
    }
    return it->second;
}


static inline i64 third_vertex(i64 a, i64 b, i64 c, i64 r0, i64 r1)
{
    if (a != r0 && a != r1) return a;
    if (b != r0 && b != r1) return b;
    if (c != r0 && c != r1) return c;
    throw std::runtime_error("could not determine opposite vertex");
}


static inline void push_child(
    std::vector<std::array<i64, 3>>& new_cells,
    std::vector<std::array<i64, 2>>& new_refedges,
    std::vector<i64>& new_celllabels,
    std::vector<i64>& parent_cell_of_child,
    i64 x0, i64 x1, i64 x2,
    i64 e0, i64 e1,
    i64 label,
    i64 parent
)
{
    new_cells.push_back({x0, x1, x2});
    new_refedges.push_back({e0, e1});
    new_celllabels.push_back(label);
    parent_cell_of_child.push_back(parent);
}
/*-----------------------------------------------------------*/
py::dict refine_cells_nvb(
    py::array_t<int> cells,
    py::array_t<int> refedges,
    py::array_t<long long> marked_edge_keys,
    py::array_t<long long> edgekey_to_mid_keys,
    py::array_t<int> edgekey_to_mid_vals,
    py::array_t<int> old_celllabels,
    int max_new_cells,
    int nkey
)
{
    auto C  = cells.unchecked<2>();
    auto RE = refedges.unchecked<2>();
    auto MK = marked_edge_keys.unchecked<1>();
    auto KM = edgekey_to_mid_keys.unchecked<1>();
    auto VM = edgekey_to_mid_vals.unchecked<1>();
    auto CL = old_celllabels.unchecked<1>();

    const py::ssize_t ncells = C.shape(0);

    std::unordered_set<long long> marked;
    marked.reserve(static_cast<std::size_t>(MK.shape(0)) * 2);

    for (py::ssize_t i = 0; i < MK.shape(0); ++i) {
        marked.insert(MK(i));
    }

    std::unordered_map<long long, int> key_to_mid;
    key_to_mid.reserve(static_cast<std::size_t>(KM.shape(0)) * 2);

    for (py::ssize_t i = 0; i < KM.shape(0); ++i) {
        key_to_mid[KM(i)] = VM(i);
    }

    py::array_t<int> new_cells({max_new_cells, 3});
    py::array_t<int> new_refedges({max_new_cells, 2});
    py::array_t<int> new_celllabels(max_new_cells);
    py::array_t<int> parent_cell_of_child(max_new_cells);

    auto NC = new_cells.mutable_unchecked<2>();
    auto NR = new_refedges.mutable_unchecked<2>();
    auto NL = new_celllabels.mutable_unchecked<1>();
    auto PC = parent_cell_of_child.mutable_unchecked<1>();

    struct Item {
        int a, b, c;
        int r0, r1;
    };

    std::vector<Item> stack;
    stack.reserve(16);

    int pos = 0;

    for (py::ssize_t icell = 0; icell < ncells; ++icell) {
        const int parent = static_cast<int>(icell);
        const int label = CL(icell);

        stack.clear();

        stack.push_back({
            C(icell, 0), C(icell, 1), C(icell, 2),
            RE(icell, 0), RE(icell, 1)
        });

        while (!stack.empty()) {
            Item it = stack.back();
            stack.pop_back();

            long long x = it.r0;
            long long y = it.r1;

            long long key;
            if (x < y) key = x * static_cast<long long>(nkey) + y;
            else       key = y * static_cast<long long>(nkey) + x;

            if (marked.find(key) == marked.end()) {
                if (pos >= max_new_cells) {
                    throw std::runtime_error("max_new_cells too small in refine_cells_nvb");
                }

                NC(pos, 0) = it.a;
                NC(pos, 1) = it.b;
                NC(pos, 2) = it.c;

                NR(pos, 0) = it.r0;
                NR(pos, 1) = it.r1;

                NL(pos) = label;
                PC(pos) = parent;

                ++pos;
                continue;
            }

            auto found = key_to_mid.find(key);
            if (found == key_to_mid.end()) {
                throw std::runtime_error("marked edge has no midpoint in refine_cells_nvb");
            }

            const int m = found->second;

            int z;
            if (it.a != it.r0 && it.a != it.r1) {
                z = it.a;
            } else if (it.b != it.r0 && it.b != it.r1) {
                z = it.b;
            } else {
                z = it.c;
            }

            stack.push_back({z, m, it.r1, z, it.r1});
            stack.push_back({z, it.r0, m, z, it.r0});
        }
    }

    py::dict result;
    result["cells"] = new_cells.attr("__getitem__")(py::slice(0, pos, 1));
    result["refedges"] = new_refedges.attr("__getitem__")(py::slice(0, pos, 1));
    result["celllabels"] = new_celllabels.attr("__getitem__")(py::slice(0, pos, 1));
    result["parent_cell_of_child"] = parent_cell_of_child.attr("__getitem__")(py::slice(0, pos, 1));
    result["nnew"] = pos;

    return result;
}


/*-----------------------------------------------------------*/
py::dict refine_nvb(
    py::array_t<double, py::array::c_style | py::array::forcecast> points_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> cells_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> refedges_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> celllabels_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> marked_in
)
{
    auto P = points_in.unchecked<2>();
    auto C = cells_in.unchecked<2>();
    auto R = refedges_in.unchecked<2>();
    auto L = celllabels_in.unchecked<1>();
    auto M = marked_in.unchecked<1>();

    const i64 npoints = P.shape(0);
    const i64 dim     = P.shape(1);
    const i64 ncells  = C.shape(0);

    if (C.shape(1) != 3) {
        throw std::runtime_error("refine_nvb currently expects triangular cells");
    }
    if (R.shape(0) != ncells || R.shape(1) != 2) {
        throw std::runtime_error("refedges must have shape (ncells, 2)");
    }
    if (L.shape(0) != ncells) {
        throw std::runtime_error("celllabels must have length ncells");
    }

    const i64 nkey = 4 * npoints + 4 * ncells + 10;

    // ------------------------------------------------------------
    // 1. Marked reference edges.
    // For now: mark the reference edge of every marked cell.
    // Closure can be added next.
    // ------------------------------------------------------------
    std::unordered_map<i64, bool> marked_edges;
    marked_edges.reserve(4 * M.shape(0) + 16);

    for (i64 k = 0; k < M.shape(0); ++k) {
        const i64 icell = M(k);
        if (icell < 0 || icell >= ncells) {
            throw std::runtime_error("marked cell index out of range");
        }
        const i64 a = R(icell, 0);
        const i64 b = R(icell, 1);
        marked_edges[edge_key(a, b, nkey)] = true;
    }

    // ------------------------------------------------------------
    // 2. Midpoints for marked edges.
    // ------------------------------------------------------------
    std::unordered_map<i64, i64> edge_to_mid;
    edge_to_mid.reserve(marked_edges.size() * 2 + 16);

    std::vector<std::vector<double>> new_points;
    new_points.reserve(npoints + marked_edges.size());

    for (i64 i = 0; i < npoints; ++i) {
        std::vector<double> x(dim);
        for (i64 d = 0; d < dim; ++d) x[d] = P(i, d);
        new_points.push_back(std::move(x));
    }

    for (const auto& kv : marked_edges) {
        const i64 key = kv.first;
        const i64 a = key / nkey;
        const i64 b = key % nkey;

        const i64 mid = static_cast<i64>(new_points.size());
        std::vector<double> xm(dim);
        for (i64 d = 0; d < dim; ++d) {
            xm[d] = 0.5 * (P(a, d) + P(b, d));
        }

        new_points.push_back(std::move(xm));
        edge_to_mid[key] = mid;
    }

    // ------------------------------------------------------------
    // 3. Refine cells.
    // This is the first simple version:
    //   - if refedge is marked: bisect;
    //   - otherwise keep the cell.
    // Full bisec12/bisec13/bisec123 comes next.
    // ------------------------------------------------------------
    std::vector<std::array<i64, 3>> new_cells;
    std::vector<std::array<i64, 2>> new_refedges;
    std::vector<i64> new_celllabels;
    std::vector<i64> parent_cell_of_child;

    new_cells.reserve(2 * ncells);
    new_refedges.reserve(2 * ncells);
    new_celllabels.reserve(2 * ncells);
    parent_cell_of_child.reserve(2 * ncells);

    for (i64 icell = 0; icell < ncells; ++icell) {
    const i64 a = C(icell, 0);
    const i64 b = C(icell, 1);
    const i64 c = C(icell, 2);

    const i64 r0 = R(icell, 0);
    const i64 r1 = R(icell, 1);

    const i64 z = third_vertex(a, b, c, r0, r1);

    // Local NVB edges:
    // e0 = reference edge
    // e1 = edge (r1, z)
    // e2 = edge (z, r0)
    const bool m0 = has_key(edge_to_mid, r0, r1, nkey);
    const bool m1 = has_key(edge_to_mid, r1, z,  nkey);
    const bool m2 = has_key(edge_to_mid, z,  r0, nkey);

    const i64 label = L(icell);

    if (!m0 && !m1 && !m2) {
        push_child(
            new_cells, new_refedges, new_celllabels, parent_cell_of_child,
            a, b, c,
            r0, r1,
            label, icell
        );
    }

    else if (m0 && !m1 && !m2) {
        // bisec1
        const i64 q0 = mid_of(edge_to_mid, r0, r1, nkey);

        push_child(
            new_cells, new_refedges, new_celllabels, parent_cell_of_child,
            z, r0, q0,
            z, r0,
            label, icell
        );

        push_child(
            new_cells, new_refedges, new_celllabels, parent_cell_of_child,
            z, q0, r1,
            r1, z,
            label, icell
        );
    }

    else if (m0 && m1 && !m2) {
        // bisec12: edges (r0,r1) and (r1,z)
        const i64 q0 = mid_of(edge_to_mid, r0, r1, nkey);
        const i64 q1 = mid_of(edge_to_mid, r1, z,  nkey);

        push_child(
            new_cells, new_refedges, new_celllabels, parent_cell_of_child,
            r1, q1, q0,
            q1, q0,
            label, icell
        );

        push_child(
            new_cells, new_refedges, new_celllabels, parent_cell_of_child,
            q1, z, r0,
            z, r0,
            label, icell
        );

        push_child(
            new_cells, new_refedges, new_celllabels, parent_cell_of_child,
            q1, r0, q0,
            r0, q1,
            label, icell
        );
    }

    else if (m0 && !m1 && m2) {
        // bisec13: edges (r0,r1) and (z,r0)
        const i64 q0 = mid_of(edge_to_mid, r0, r1, nkey);
        const i64 q2 = mid_of(edge_to_mid, z,  r0, nkey);

        push_child(
            new_cells, new_refedges, new_celllabels, parent_cell_of_child,
            r0, q0, q2,
            q0, q2,
            label, icell
        );

        push_child(
            new_cells, new_refedges, new_celllabels, parent_cell_of_child,
            q0, r1, z,
            r1, z,
            label, icell
        );

        push_child(
            new_cells, new_refedges, new_celllabels, parent_cell_of_child,
            q0, z, q2,
            z, q0,
            label, icell
        );
    }

    else if (m0 && m1 && m2) {
        // bisec123
        const i64 q0 = mid_of(edge_to_mid, r0, r1, nkey);
        const i64 q1 = mid_of(edge_to_mid, r1, z,  nkey);
        const i64 q2 = mid_of(edge_to_mid, z,  r0, nkey);

        push_child(
            new_cells, new_refedges, new_celllabels, parent_cell_of_child,
            r0, q0, q2,
            q0, q2,
            label, icell
        );

        push_child(
            new_cells, new_refedges, new_celllabels, parent_cell_of_child,
            q0, r1, q1,
            r1, q1,
            label, icell
        );

        push_child(
            new_cells, new_refedges, new_celllabels, parent_cell_of_child,
            q2, q1, z,
            q1, z,
            label, icell
        );

        push_child(
            new_cells, new_refedges, new_celllabels, parent_cell_of_child,
            q0, q1, q2,
            q1, q2,
            label, icell
        );
    }

    else {
        throw std::runtime_error(
            "NVB closure violated: non-reference edge marked without reference edge"
        );
    }
}
//    for (i64 icell = 0; icell < ncells; ++icell) {
//        const i64 a = C(icell, 0);
//        const i64 b = C(icell, 1);
//        const i64 c = C(icell, 2);
//
//        const i64 r0 = R(icell, 0);
//        const i64 r1 = R(icell, 1);
//        const i64 kref = edge_key(r0, r1, nkey);
//
//        auto it = edge_to_mid.find(kref);
//
//        if (it == edge_to_mid.end()) {
//            new_cells.push_back({a, b, c});
//            new_refedges.push_back({r0, r1});
//            new_celllabels.push_back(L(icell));
//            parent_cell_of_child.push_back(icell);
//            continue;
//        }
//
//        const i64 m = it->second;
//
//        // Opposite vertex to reference edge.
//        i64 z = -1;
//        if (a != r0 && a != r1) z = a;
//        if (b != r0 && b != r1) z = b;
//        if (c != r0 && c != r1) z = c;
//
//        if (z < 0) {
//            throw std::runtime_error("could not determine opposite vertex");
//        }
//
//        // Children: (z, r0, m), (z, m, r1)
//        new_cells.push_back({z, r0, m});
//        new_refedges.push_back({z, r0});
//
//        new_cells.push_back({z, m, r1});
//        new_refedges.push_back({r1, z});
//
//        new_celllabels.push_back(L(icell));
//        new_celllabels.push_back(L(icell));
//
//        parent_cell_of_child.push_back(icell);
//        parent_cell_of_child.push_back(icell);
//    }

    // ------------------------------------------------------------
    // 4. Export arrays.
    // ------------------------------------------------------------
    py::array_t<double> points_out({static_cast<py::ssize_t>(new_points.size()),
                                    static_cast<py::ssize_t>(dim)});
    auto PO = points_out.mutable_unchecked<2>();

    for (i64 i = 0; i < static_cast<i64>(new_points.size()); ++i) {
        for (i64 d = 0; d < dim; ++d) {
            PO(i, d) = new_points[i][d];
        }
    }

py::array_t<i64> cells_out(
    std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(new_cells.size()),
        static_cast<py::ssize_t>(3)
    }
);

py::array_t<i64> refedges_out(
    std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(new_refedges.size()),
        static_cast<py::ssize_t>(2)
    }
);
    auto CO = cells_out.mutable_unchecked<2>();

    auto RO = refedges_out.mutable_unchecked<2>();

    py::array_t<i64> labels_out(static_cast<py::ssize_t>(new_celllabels.size()));
    auto LO = labels_out.mutable_unchecked<1>();

    py::array_t<i64> parent_out(static_cast<py::ssize_t>(parent_cell_of_child.size()));
    auto PA = parent_out.mutable_unchecked<1>();

    for (i64 i = 0; i < static_cast<i64>(new_cells.size()); ++i) {
        CO(i, 0) = new_cells[i][0];
        CO(i, 1) = new_cells[i][1];
        CO(i, 2) = new_cells[i][2];

        RO(i, 0) = new_refedges[i][0];
        RO(i, 1) = new_refedges[i][1];

        LO(i) = new_celllabels[i];
        PA(i) = parent_cell_of_child[i];
    }

    py::dict out;
    out["points"] = points_out;
    out["cells"] = cells_out;
    out["refedges"] = refedges_out;
    out["celllabels"] = labels_out;
    out["parent_cell_of_child"] = parent_out;

    return out;
}