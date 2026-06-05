# FEM2D/mesh/refinement_info.py

from dataclasses import dataclass
import numpy as np


@dataclass
class RefinementInfo:
    old_npoints: int
    new_npoints: int
    old_ncells: int
    new_ncells: int

    midpoint_parents: dict[int, tuple[int, int]]
    parent_cell_of_child: np.ndarray

    # for CR1 transfer
    old_nfaces: int | None = None
    new_nfaces: int | None = None
    old_points: np.ndarray | None = None
    old_cells: np.ndarray | None = None
    old_faces_of_cells: np.ndarray | None = None
    new_face_centers: np.ndarray | None = None
    parent_cell_of_face: np.ndarray | None = None