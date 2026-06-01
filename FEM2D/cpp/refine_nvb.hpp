#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::dict unique_edges(py::array_t<int> edges);

py::dict build_midpoints(
    py::array_t<double> points,
    py::array_t<int> edges
);

py::dict refine_cells_nvb(
    py::array_t<int> cells,
    py::array_t<int> refedges,
    py::array_t<long long> marked_edge_keys,
    py::array_t<long long> edgekey_to_mid_keys,
    py::array_t<int> edgekey_to_mid_vals,
    py::array_t<int> old_celllabels,
    int max_new_cells,
    int nkey
);