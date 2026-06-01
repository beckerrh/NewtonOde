#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <vector>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

using i64 = std::int64_t;

static inline i64 edge_key(i64 a, i64 b)
{
    if (a > b) std::swap(a, b);
    return (a << 32) ^ b;
}

py::dict construct_faces_from_cells(
    py::array_t<i64, py::array::c_style | py::array::forcecast> cells_in,
    bool build_edge2face
)
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

    std::unordered_map<i64, i64> edge_to_face;
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

    py::dict out;
    out["faces"] = faces_out;
    out["faces_of_cells"] = foc_out;
    out["cells_of_faces"] = cof_out;

    if (build_edge2face) {
        py::dict edge2face;

        for (i64 iface = 0; iface < nfaces; ++iface) {
            py::tuple e(2);
            e[0] = faces[iface][0];
            e[1] = faces[iface][1];
            edge2face[e] = iface;
        }

        out["edge2face"] = edge2face;
    }

    return out;
}