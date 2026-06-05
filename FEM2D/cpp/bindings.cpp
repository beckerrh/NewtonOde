#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>

#include "refine_nvb.hpp"
#include "topology.hpp"
#include "geometry.hpp"
#include "mesh_rebuild.hpp"

namespace py = pybind11;


PYBIND11_MODULE(_meshcpp, m)
{
    m.def("rebuild_mesh_2d", &rebuild_mesh_2d);
    m.def(
        "construct_faces_from_cells",
        &construct_faces_from_cells,
        py::arg("cells")
    );
    m.def("refine_cells_nvb",
    &refine_cells_nvb,
    "Refine cells by recursive NVB using precomputed marked edge keys and midpoint ids");
    m.def(
    "cell_ref_faces_from_refedges",
    &cell_ref_faces_from_refedges,
    "Map each cell reference edge to its global face index"
    );
    m.def(
    "close_marked_faces_nvb",
    &close_marked_faces_nvb,
    "NVB closure using face adjacency"
    );
    m.def(
    "boundary_edges_to_faces",
    &boundary_edges_to_faces,
    py::arg("faces"),
    py::arg("boundary_edges")
    );
    m.def("boundary_edges_to_faces_all", &boundary_edges_to_faces_all);
    m.def(
    "construct_geometry_2d",
    &construct_geometry_2d,
    py::arg("points"),
    py::arg("cells"),
    py::arg("faces"),
    py::arg("faces_of_cells"),
    py::arg("cells_of_faces")
    );
}
