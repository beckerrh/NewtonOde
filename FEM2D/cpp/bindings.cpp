#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>

namespace py = pybind11;

py::dict construct_faces_from_cells(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cells,
    bool build_edge2face
);

py::dict refine_nvb(
    py::array_t<double, py::array::c_style | py::array::forcecast> points_in,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cells_in,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> refedges_in,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> celllabels_in,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> marked_in
);

PYBIND11_MODULE(_mesh_cpp, m)
{
    m.def(
        "construct_faces_from_cells",
        &construct_faces_from_cells,
        py::arg("cells"),
        py::arg("build_edge2face") = true
    );
    m.def("refine_nvb", &refine_nvb, "Newest-vertex bisection refinement kernel");
}
