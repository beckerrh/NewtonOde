# backend_dispatch.py

from FEM2D.mesh.backend import cpp_backend
from FEM2D.mesh import topology, geometry

def rebuild_mesh(mesh, timer=None):
    if cpp_backend is not None:
        return rebuild_mesh_cpp(mesh, timer=timer)
    return rebuild_mesh_python(mesh, timer=timer)


def refine_nvb(mesh, marked, debug=False, timer=None):
    if cpp_backend is not None:
        from FEM2D.mesh.refine_nvb import refine_nvb_cpp
        mesh2, info = refine_nvb_cpp(mesh, marked, debug=debug, timer=timer)
    else:
        from FEM2D.mesh.refine_nvb import refine_nvb_python
        mesh2, info = refine_nvb_python(mesh, marked, debug=debug, timer=timer)
    from FEM2D.mesh.geometry import correct_boundary_geometry
    mesh2 = correct_boundary_geometry(mesh, mesh2)
    return mesh2, info

def rebuild_mesh_python(mesh, timer=None):
    topology.construct_faces_from_cells_python(mesh)
    geometry.construct_centers(mesh)
    geometry.construct_normals_and_volumes(mesh)

    mesh.ncells = mesh.topology.cells.shape[0]
    mesh.nfaces = mesh.topology.faces.shape[0]

    topology.construct_inner_faces(mesh)

def rebuild_mesh_cpp(mesh, timer=None):
    r = cpp_backend.rebuild_mesh_2d(
        mesh.geometry.points,
        mesh.topology.cells,
    )

    mesh.topology.faces = r["faces"]
    mesh.topology.faces_of_cells = r["faces_of_cells"]
    mesh.topology.cells_of_faces = r["cells_of_faces"]

    mesh.geometry.cell_centers = r["cell_centers"]
    mesh.geometry.face_centers = r["face_centers"]
    mesh.geometry.normals = r["normals"]
    mesh.geometry.cell_volumes = r["cell_volumes"]
    mesh.sigma = r["sigma"]

    mesh.ncells = mesh.topology.cells.shape[0]
    mesh.nfaces = mesh.topology.faces.shape[0]
    mesh.edge2face = {
        (int(a), int(b)): int(i)
        for i, (a, b) in enumerate(mesh.topology.faces)
    }

    topology.construct_inner_faces(mesh)