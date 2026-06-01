from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(root))

import numpy as np
import FEM2D._mesh_cpp as cpp

points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
], dtype=float)

cells = np.array([
    [0, 1, 2],
], dtype=np.int64)

refedges = np.array([
    [1, 2],
], dtype=np.int64)

celllabels = np.array([0], dtype=np.int64)
marked = np.array([0], dtype=np.int64)

r = cpp.refine_nvb(points, cells, refedges, celllabels, marked)

for k, v in r.items():
    print(k, type(v))
    print(v)