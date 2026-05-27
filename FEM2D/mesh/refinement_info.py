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
    child_cells_of_parent: dict[int, list[int]]
    parent_cell_of_child: np.ndarray