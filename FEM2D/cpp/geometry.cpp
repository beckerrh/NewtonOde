#include "geometry.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

GeometryData2D build_geometry_2d(
    py::array_t<double, py::array::c_style | py::array::forcecast> points_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> cells_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> faces_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> faces_of_cells_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> cells_of_faces_in
)
{
    // Put the OLD body of construct_geometry_2d here.
    // But instead of returning py::dict, fill GeometryData2D g.

    GeometryData2D g;

    /*
       existing geometry allocation/computation, producing:

       cell_centers
       face_centers
       normals
       cell_volumes
       sigma
    */
    auto P   = points_in.unchecked<2>();
    auto C   = cells_in.unchecked<2>();
    auto F   = faces_in.unchecked<2>();
    auto FOC = faces_of_cells_in.unchecked<2>();
    auto COF = cells_of_faces_in.unchecked<2>();

    const i64 npoints = P.shape(0);
    const i64 dim     = P.shape(1);
    const i64 ncells  = C.shape(0);
    const i64 nfaces  = F.shape(0);

    if (dim < 2) {
        throw std::runtime_error("construct_geometry_2d expects points with at least two coordinates");
    }
    if (C.shape(1) != 3) {
        throw std::runtime_error("construct_geometry_2d expects triangular cells");
    }
    if (F.shape(1) != 2) {
        throw std::runtime_error("construct_geometry_2d expects edges/faces of shape (nfaces,2)");
    }
    if (FOC.shape(0) != ncells || FOC.shape(1) != 3) {
        throw std::runtime_error("faces_of_cells has wrong shape");
    }
    if (COF.shape(0) != nfaces || COF.shape(1) != 2) {
        throw std::runtime_error("cells_of_faces has wrong shape");
    }

    py::array_t<double> cell_centers_out(
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(ncells),
            static_cast<py::ssize_t>(2)
        }
    );
    py::array_t<double> face_centers_out(
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(nfaces),
            static_cast<py::ssize_t>(2)
        }
    );

    py::array_t<double> normals_out(
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(nfaces),
            static_cast<py::ssize_t>(3)
        }
    );

    py::array_t<double> cell_volumes_out(
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(ncells)
        }
    );

    py::array_t<i64> sigma_out(
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(ncells),
            static_cast<py::ssize_t>(3)
        }
    );
    auto CC = cell_centers_out.mutable_unchecked<2>();
    auto FC = face_centers_out.mutable_unchecked<2>();
    auto N  = normals_out.mutable_unchecked<2>();
    auto V  = cell_volumes_out.mutable_unchecked<1>();
    auto S  = sigma_out.mutable_unchecked<2>();

    // ------------------------------------------------------------
    // Cell centers and volumes.
    // ------------------------------------------------------------
    for (i64 icell = 0; icell < ncells; ++icell) {
        const i64 a = C(icell, 0);
        const i64 b = C(icell, 1);
        const i64 c = C(icell, 2);

        const double xa = P(a, 0);
        const double ya = P(a, 1);
        const double xb = P(b, 0);
        const double yb = P(b, 1);
        const double xc = P(c, 0);
        const double yc = P(c, 1);

        CC(icell, 0) = (xa + xb + xc) / 3.0;
        CC(icell, 1) = (ya + yb + yc) / 3.0;

        const double dx1 = xb - xa;
        const double dy1 = yb - ya;
        const double dx2 = xc - xa;
        const double dy2 = yc - ya;

        V(icell) = 0.5 * std::abs(dx1 * dy2 - dx2 * dy1);
    }

    // ------------------------------------------------------------
    // Face centers and initial normals.
    // Normal convention before orientation:
    // edge (a,b) -> normal (-(yb-ya), xb-xa, 0).
    // ------------------------------------------------------------
    for (i64 iface = 0; iface < nfaces; ++iface) {
        const i64 a = F(iface, 0);
        const i64 b = F(iface, 1);

        const double xa = P(a, 0);
        const double ya = P(a, 1);
        const double xb = P(b, 0);
        const double yb = P(b, 1);

        FC(iface, 0) = 0.5 * (xa + xb);
        FC(iface, 1) = 0.5 * (ya + yb);

        N(iface, 0) = -(yb - ya);
        N(iface, 1) =  (xb - xa);
        N(iface, 2) = 0.0;
    }

    // ------------------------------------------------------------
    // sigma[icell, iloc].
    //
    // Same as Python:
    // sigma = 2 * (cells_of_faces[faces_of_cells,0] == icell) - 1
    // ------------------------------------------------------------
    for (i64 icell = 0; icell < ncells; ++icell) {
        for (i64 iloc = 0; iloc < 3; ++iloc) {
            const i64 iface = FOC(icell, iloc);
            S(icell, iloc) = (COF(iface, 0) == icell) ? 1 : -1;
        }
    }

    // ------------------------------------------------------------
    // Orient normals.
    //
    // Boundary face: outward from adjacent cell.
    // Interior face: from cell 0 to cell 1.
    // ------------------------------------------------------------
    for (i64 iface = 0; iface < nfaces; ++iface) {
        const i64 c0 = COF(iface, 0);
        const i64 c1 = COF(iface, 1);

        double vx;
        double vy;

        if (c1 < 0) {
            // Boundary face.
            vx = FC(iface, 0) - CC(c0, 0);
            vy = FC(iface, 1) - CC(c0, 1);
        } else {
            // Interior face.
            vx = CC(c1, 0) - CC(c0, 0);
            vy = CC(c1, 1) - CC(c0, 1);
        }

        const double dot = N(iface, 0) * vx + N(iface, 1) * vy;

        if (dot < 0.0) {
            N(iface, 0) = -N(iface, 0);
            N(iface, 1) = -N(iface, 1);
        }
    }

    g.cell_centers = cell_centers_out;
    g.face_centers = face_centers_out;
    g.normals = normals_out;
    g.cell_volumes = cell_volumes_out;
    g.sigma = sigma_out;

    return g;
}


py::dict construct_geometry_2d(
    py::array_t<double, py::array::c_style | py::array::forcecast> points_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> cells_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> faces_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> faces_of_cells_in,
    py::array_t<i64, py::array::c_style | py::array::forcecast> cells_of_faces_in
)
{
    GeometryData2D g = build_geometry_2d(
        points_in,
        cells_in,
        faces_in,
        faces_of_cells_in,
        cells_of_faces_in
    );

    py::dict r;
    r["cell_centers"] = g.cell_centers;
    r["face_centers"] = g.face_centers;
    r["normals"] = g.normals;
    r["cell_volumes"] = g.cell_volumes;
    r["sigma"] = g.sigma;
    return r;
}