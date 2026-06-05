#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>

namespace py = pybind11;

using i64 = std::int64_t;

struct TopologyData2D {
    py::array_t<i64> faces;
    py::array_t<i64> faces_of_cells;
    py::array_t<i64> cells_of_faces;
};

struct GeometryData2D {
    py::array_t<double> cell_centers;
    py::array_t<double> face_centers;
    py::array_t<double> normals;
    py::array_t<double> cell_volumes;
    py::array_t<i64> sigma;
};