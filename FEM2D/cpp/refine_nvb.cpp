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
using u64 = std::uint64_t;

static inline i64 edge_key(i64 a, i64 b, i64 nkey)
{
return (static_cast<std::uint64_t>(a) << 32)
     | static_cast<std::uint64_t>(b);
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
    marked.reserve(static_cast<std::size_t>(MK.shape(0)) * 2 + 16);

    for (py::ssize_t i = 0; i < MK.shape(0); ++i) {
        marked.insert(MK(i));
    }

    std::unordered_map<long long, int> key_to_mid;
    key_to_mid.reserve(static_cast<std::size_t>(KM.shape(0)) * 2 + 16);

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
    stack.reserve(32);

    int pos = 0;

    for (py::ssize_t icell = 0; icell < ncells; ++icell) {
        const int parent = static_cast<int>(icell);
        const int label = CL(icell);

        stack.clear();

        stack.push_back({
            C(icell, 0), C(icell, 1), C(icell, 2),
            RE(icell, 0), RE(icell, 1)
        });

        const int first_child = pos;

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
                    throw std::runtime_error(
                        "max_new_cells too small in refine_cells_nvb"
                    );
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
                throw std::runtime_error(
                    "marked edge has no midpoint in refine_cells_nvb"
                );
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

        if (pos == first_child) {
            throw std::runtime_error(
                "internal error: cell produced no children in refine_cells_nvb"
            );
        }
    }

    py::dict result;
    result["cells"] = new_cells.attr("__getitem__")(py::slice(0, pos, 1));
    result["refedges"] = new_refedges.attr("__getitem__")(py::slice(0, pos, 1));
    result["celllabels"] = new_celllabels.attr("__getitem__")(py::slice(0, pos, 1));
    result["parent_cell_of_child"] =
        parent_cell_of_child.attr("__getitem__")(py::slice(0, pos, 1));
    result["nnew"] = pos;

    return result;
}
