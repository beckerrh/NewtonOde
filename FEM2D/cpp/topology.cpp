#include "topology.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <vector>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

using u64 = std::uint64_t;

static inline i64 edge_key(i64 a, i64 b)
{
    return (static_cast<std::uint64_t>(a) << 32)
     | static_cast<std::uint64_t>(b);
    if (a > b) std::swap(a, b);
    return (a << 32) ^ b;
}

/*-----------------------------------------------------------*/
py::dict boundary_edges_to_faces_all(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> faces_in,
    py::dict boundary_edge_labels
)
{
    auto faces = faces_in.unchecked<2>();
    const ssize_t nfaces = faces.shape(0);

    std::unordered_map<std::int64_t, std::int64_t> edge_to_face;
    edge_to_face.reserve(static_cast<std::size_t>(2 * nfaces));

    // Need nkey larger than max point index.
    std::int64_t maxv = 0;
    for (ssize_t i = 0; i < nfaces; ++i) {
        maxv = std::max(maxv, faces(i, 0));
        maxv = std::max(maxv, faces(i, 1));
    }
    const std::int64_t nkey = maxv + 1;

    for (ssize_t i = 0; i < nfaces; ++i) {
        std::int64_t a = faces(i, 0);
        std::int64_t b = faces(i, 1);
        if (a > b) std::swap(a, b);
        edge_to_face[a * nkey + b] = static_cast<std::int64_t>(i);
    }

    py::dict out;

    for (auto item : boundary_edge_labels) {
        py::object label = py::reinterpret_borrow<py::object>(item.first);

        py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> edges_arr =
            py::cast<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>>(item.second);

        auto edges = edges_arr.unchecked<2>();
        const ssize_t nedges = edges.shape(0);

        std::vector<std::int64_t> ids;
        ids.reserve(static_cast<std::size_t>(nedges));

        for (ssize_t j = 0; j < nedges; ++j) {
            std::int64_t a = edges(j, 0);
            std::int64_t b = edges(j, 1);
            if (a > b) std::swap(a, b);

            const auto it = edge_to_face.find(a * nkey + b);
            if (it != edge_to_face.end()) {
                ids.push_back(it->second);
            }
        }

        py::array_t<std::int64_t> ids_arr(ids.size());
        auto r = ids_arr.mutable_unchecked<1>();
        for (ssize_t k = 0; k < static_cast<ssize_t>(ids.size()); ++k) {
            r(k) = ids[static_cast<std::size_t>(k)];
        }

        out[label] = ids_arr;
    }

    return out;
}
/*-----------------------------------------------------------*/
py::array_t<i64> boundary_edges_to_faces(
    py::array_t<i64, py::array::c_style | py::array::forcecast> faces_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> edges_in
)
{
    auto F = faces_in.unchecked<2>();
    auto E = edges_in.unchecked<2>();

    const i64 nfaces = F.shape(0);
    const i64 nedges = E.shape(0);

    if (F.shape(1) != 2 || E.shape(1) != 2) {
        throw std::runtime_error("boundary_edges_to_faces expects (n,2) arrays");
    }

    std::unordered_map<i64, i64> edge_to_face;
    edge_to_face.reserve(static_cast<std::size_t>(2 * nfaces));

    for (i64 iface = 0; iface < nfaces; ++iface) {
        i64 a = F(iface, 0);
        i64 b = F(iface, 1);
        if (a > b) std::swap(a, b);

        edge_to_face[edge_key(a, b)] = iface;
    }

    std::vector<i64> ids;
    ids.reserve(static_cast<std::size_t>(nedges));

    for (i64 i = 0; i < nedges; ++i) {
        i64 a = E(i, 0);
        i64 b = E(i, 1);
        if (a > b) std::swap(a, b);

        auto it = edge_to_face.find(edge_key(a, b));
        if (it != edge_to_face.end()) {
            ids.push_back(it->second);
        }
    }

    py::array_t<i64> out(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(ids.size())}
    );

    auto O = out.mutable_unchecked<1>();
    for (i64 i = 0; i < static_cast<i64>(ids.size()); ++i) {
        O(i) = ids[static_cast<std::size_t>(i)];
    }

    return out;
}
/*-----------------------------------------------------------*/
py::array_t<i64> cell_ref_faces_from_refedges(
    py::array_t<i64, py::array::c_style | py::array::forcecast> faces_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> refedges_in
)
{
    auto F  = faces_in.unchecked<2>();
    auto RE = refedges_in.unchecked<2>();

    const i64 nfaces = F.shape(0);
    const i64 ncells = RE.shape(0);

    std::unordered_map<i64, i64> edge_to_face;
    edge_to_face.reserve(2 * nfaces);

    for (i64 iface = 0; iface < nfaces; ++iface) {
        i64 a = F(iface, 0);
        i64 b = F(iface, 1);
        if (a > b) std::swap(a, b);

        edge_to_face[edge_key(a, b)] = iface;
    }

    py::array_t<i64> out(ncells);
    auto O = out.mutable_unchecked<1>();

    for (i64 icell = 0; icell < ncells; ++icell) {
        i64 a = RE(icell, 0);
        i64 b = RE(icell, 1);
        if (a > b) std::swap(a, b);

        auto it = edge_to_face.find(edge_key(a, b));
        if (it == edge_to_face.end()) {
            throw std::runtime_error("reference edge not found in faces");
        }

        O(icell) = it->second;
    }

    return out;
}
/*-----------------------------------------------------------*/
py::dict construct_faces_from_cells(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cells
)
{
    TopologyData2D t = build_topology_2d(cells);

    py::dict r;
    r["faces"] = t.faces;
    r["faces_of_cells"] = t.faces_of_cells;
    r["cells_of_faces"] = t.cells_of_faces;
    return r;
}
/*-----------------------------------------------------------*/
TopologyData2D build_topology_2d(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cells_in)
{
    auto C = cells_in.unchecked<2>();

    const i64 ncells = C.shape(0);
    const i64 nloc   = C.shape(1);

    if (nloc != 3) {
        throw std::runtime_error("construct_faces_from_cells currently expects triangular cells");
    }

    std::vector<std::array<i64, 2>> faces;
    faces.reserve(3 * ncells);

    std::vector<std::array<i64, 3>> faces_of_cells;
    faces_of_cells.resize(ncells);

    std::vector<std::array<i64, 2>> cells_of_faces;


    std::unordered_map<u64, i64> edge_to_face;
//    std::unordered_map<i64, i64> edge_to_face;
    edge_to_face.reserve(3 * ncells);

    for (i64 icell = 0; icell < ncells; ++icell) {
        const i64 a = C(icell, 0);
        const i64 b = C(icell, 1);
        const i64 c = C(icell, 2);

        const i64 e0[2] = {b, c};
        const i64 e1[2] = {c, a};
        const i64 e2[2] = {a, b};

        const i64 edges[3][2] = {
            {e0[0], e0[1]},
            {e1[0], e1[1]},
            {e2[0], e2[1]},
        };

        for (i64 iloc = 0; iloc < 3; ++iloc) {
            i64 u = edges[iloc][0];
            i64 v = edges[iloc][1];

            i64 su = u;
            i64 sv = v;
            if (su > sv) std::swap(su, sv);

            const i64 key = edge_key(su, sv);

            auto it = edge_to_face.find(key);

            if (it == edge_to_face.end()) {
                const i64 iface = static_cast<i64>(faces.size());

                edge_to_face[key] = iface;
                faces.push_back({su, sv});
                cells_of_faces.push_back({icell, -1});
                faces_of_cells[icell][iloc] = iface;
            } else {
                const i64 iface = it->second;

                if (cells_of_faces[iface][1] != -1) {
                    throw std::runtime_error("nonmanifold edge detected");
                }

                cells_of_faces[iface][1] = icell;
                faces_of_cells[icell][iloc] = iface;
            }
        }
    }

    const i64 nfaces = static_cast<i64>(faces.size());

    py::array_t<i64> faces_out(
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(nfaces),
            static_cast<py::ssize_t>(2)
        }
    );

    py::array_t<i64> foc_out(
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(ncells),
            static_cast<py::ssize_t>(3)
        }
    );

    py::array_t<i64> cof_out(
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(nfaces),
            static_cast<py::ssize_t>(2)
        }
    );

    auto F   = faces_out.mutable_unchecked<2>();
    auto FOC = foc_out.mutable_unchecked<2>();
    auto COF = cof_out.mutable_unchecked<2>();

    for (i64 iface = 0; iface < nfaces; ++iface) {
        F(iface, 0) = faces[iface][0];
        F(iface, 1) = faces[iface][1];

        COF(iface, 0) = cells_of_faces[iface][0];
        COF(iface, 1) = cells_of_faces[iface][1];
    }

    for (i64 icell = 0; icell < ncells; ++icell) {
        FOC(icell, 0) = faces_of_cells[icell][0];
        FOC(icell, 1) = faces_of_cells[icell][1];
        FOC(icell, 2) = faces_of_cells[icell][2];
    }

    TopologyData2D t;

    t.faces = faces_out;
    t.faces_of_cells = foc_out;
    t.cells_of_faces = cof_out;

    return t;
}



py::dict close_marked_faces_nvb(
    py::array_t<i64, py::array::c_style | py::array::forcecast> cells_of_faces_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> cell_ref_faces_in,
    py::array_t<bool, py::array::c_style | py::array::forcecast> marked_in
)
{
    auto COF = cells_of_faces_in.unchecked<2>();
    auto CRF = cell_ref_faces_in.unchecked<1>();
    auto M   = marked_in.unchecked<1>();

    const i64 nfaces = COF.shape(0);
    const i64 ncells = CRF.shape(0);

    if (COF.shape(1) != 2) {
        throw std::runtime_error("cells_of_faces must have shape (nfaces, 2)");
    }
    if (M.shape(0) != ncells) {
        throw std::runtime_error("marked and cell_ref_faces size mismatch");
    }

    py::array_t<bool> marked_bool(ncells);
    py::array_t<bool> refine_face(nfaces);

    auto MB = marked_bool.mutable_unchecked<1>();
    auto RF = refine_face.mutable_unchecked<1>();

    std::vector<i64> queue;
    queue.resize(ncells);

    i64 head = 0;
    i64 tail = 0;

    for (i64 i = 0; i < ncells; ++i) {
        MB(i) = M(i);
        if (MB(i)) {
            queue[tail++] = i;
        }
    }

    for (i64 f = 0; f < nfaces; ++f) {
        RF(f) = false;
    }

    while (head < tail) {
        const i64 icell = queue[head++];

        const i64 iface = CRF(icell);
        if (iface < 0 || iface >= nfaces) {
            throw std::runtime_error("cell_ref_faces contains invalid face index");
        }

        RF(iface) = true;

        const i64 c0 = COF(iface, 0);
        const i64 c1 = COF(iface, 1);

        const i64 nb = (c0 == icell) ? c1 : c0;

        if (nb >= 0 && !MB(nb)) {
            MB(nb) = true;
            queue[tail++] = nb;
        }
    }

    std::vector<i64> refined;
    refined.reserve(nfaces);

    for (i64 f = 0; f < nfaces; ++f) {
        if (RF(f)) {
            refined.push_back(f);
        }
    }

    py::array_t<i64> refined_faces(
        static_cast<py::ssize_t>(refined.size())
    );
    auto RFO = refined_faces.mutable_unchecked<1>();

    for (i64 i = 0; i < static_cast<i64>(refined.size()); ++i) {
        RFO(i) = refined[i];
    }

    py::dict out;
    out["marked_bool"] = marked_bool;
    out["refine_face"] = refine_face;
    out["refined_faces"] = refined_faces;

    return out;
}