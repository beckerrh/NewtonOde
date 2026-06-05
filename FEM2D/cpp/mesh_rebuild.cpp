#include "mesh_rebuild.hpp"
#include "topology.hpp"
#include "geometry.hpp"

#include <chrono>
#include <iostream>


py::dict rebuild_mesh_2d(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<i64, py::array::c_style | py::array::forcecast> cells)
{
//    auto t0 = std::chrono::high_resolution_clock::now();

    TopologyData2D topo = build_topology_2d(cells);

//    auto t1 = std::chrono::high_resolution_clock::now();

    GeometryData2D geom = build_geometry_2d(
        points,
        cells,
        topo.faces,
        topo.faces_of_cells,
        topo.cells_of_faces
    );


//    auto t2 = std::chrono::high_resolution_clock::now();
//
//    double topo_time =
//    std::chrono::duration<double>(t1 - t0).count();
//
//    double geom_time =
//    std::chrono::duration<double>(t2 - t1).count();
//
//    std::cout
//    << "topology=" << topo_time
//    << " geometry=" << geom_time
//    << std::endl;


    py::dict r;

    r["faces"] = topo.faces;
    r["faces_of_cells"] = topo.faces_of_cells;
    r["cells_of_faces"] = topo.cells_of_faces;

    r["cell_centers"] = geom.cell_centers;
    r["face_centers"] = geom.face_centers;
    r["normals"] = geom.normals;
    r["cell_volumes"] = geom.cell_volumes;
    r["sigma"] = geom.sigma;

    return r;
}