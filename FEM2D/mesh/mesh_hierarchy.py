from dataclasses import dataclass
from FEM2D.mesh.simplex_mesh import SimplexMesh
from FEM2D.mesh.refinement_info import RefinementInfo


@dataclass
class MeshLevel:
    mesh: SimplexMesh
    level: int
    parent: "MeshLevel | None" = None
    child: "MeshLevel | None" = None
    refine_info: RefinementInfo | None = None


class MeshHierarchy:
    def __init__(self, mesh: SimplexMesh):
        self.levels = [MeshLevel(mesh=mesh, level=0)]

    @property
    def current(self):
        return self.levels[-1]

    @property
    def mesh(self):
        return self.current.mesh

    def refine_nvb(self, marked, *, timer=None, debug=False):
        mesh2, info = self.mesh.refine_nvb(
            marked,
            timer=timer,
            debug=debug,
        )

        parent = self.current
        child = MeshLevel(
            mesh=mesh2,
            level=parent.level + 1,
            parent=parent,
            refine_info=info,
        )
        parent.child = child
        self.levels.append(child)

        return mesh2, info

    def meshes(self):
        return [level.mesh for level in self.levels]

    def refinement_infos(self):
        return [level.refine_info for level in self.levels[1:]]