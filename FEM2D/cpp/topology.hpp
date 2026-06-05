#pragma once

#include "mesh_data.hpp"

TopologyData2D build_topology_2d(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cells
);





py::dict construct_faces_from_cells(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cells);

py::array_t<i64> boundary_edges_to_faces(
    py::array_t<i64, py::array::c_style | py::array::forcecast> faces_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> edges_in
);
py::dict boundary_edges_to_faces_all(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> faces_in,
    py::dict boundary_edge_labels
);


py::array_t<i64> cell_ref_faces_from_refedges(
    py::array_t<i64, py::array::c_style | py::array::forcecast> faces_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> refedges_in
);

py::dict close_marked_faces_nvb(
    py::array_t<i64, py::array::c_style | py::array::forcecast> cells_of_faces_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> cell_ref_faces_in,
    py::array_t<bool, py::array::c_style | py::array::forcecast> marked_in
);