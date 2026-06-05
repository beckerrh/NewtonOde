#pragma once

#include "mesh_data.hpp"

GeometryData2D build_geometry_2d(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cells,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> faces,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> faces_of_cells,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cells_of_faces
);

py::dict construct_geometry_2d(
    py::array_t<double, py::array::c_style | py::array::forcecast> points_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> cells_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> faces_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> faces_of_cells_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> cells_of_faces_in
);