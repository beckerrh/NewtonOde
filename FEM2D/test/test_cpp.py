from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(root))

import numpy as np
import FEM2D._refine as _refine

points = np.array([
    [0., 0.],
    [1., 0.],
    [0., 1.],
])

edges = np.array([
    [0, 1],
    [1, 2],
], dtype=np.int32)

r = _refine.build_midpoints(points, edges)

print(r["points"])
print(r["midpoint_ids"])