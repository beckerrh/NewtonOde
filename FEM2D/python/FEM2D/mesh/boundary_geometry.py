import numpy as np

def boundary_node_labels(mesh):
    node_labels = {}

    for label, faces in mesh.labels.boundary.items():
        for f in faces:
            for a in mesh.topology.faces[f]:
                node_labels.setdefault(int(a), set()).add(label)

    return node_labels

class CircleProjector:
    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

    def project(self, label, x):
        x = np.asarray(x, dtype=float)
        out = x.copy()

        d = self.center.shape[0]   # usually 2
        y = x[:d] - self.center
        r = np.linalg.norm(y)

        if r == 0.0:
            return out

        out[:d] = self.center + self.radius * y / r
        return out

    def project_many(self, label, x):
        x = np.asarray(x, dtype=float)
        out = x.copy()

        d = self.center.shape[0]
        y = x[:, :d] - self.center
        r = np.linalg.norm(y, axis=1)

        good = r > 0.0
        out[good, :d] = self.center + self.radius * y[good] / r[good, None]
        return out

class LabelBoundaryProjector:
    def __init__(self, projectors, blend_corners=False):
        self.projectors = projectors
        self.blend_corners = blend_corners

    def correct_points(self, mesh, points):
        points = points.copy()
        node_labels = boundary_node_labels(mesh)

        for node, labels in node_labels.items():
            x = points[node]

            if len(labels) == 1:
                label = next(iter(labels))
                proj = self.projectors.get(label)
                if proj is not None:
                    points[node] = proj.project(label, x)

            else:
                # corner / boundary intersection
                if not self.blend_corners:
                    continue

                xs = []
                for label in labels:
                    proj = self.projectors.get(label)
                    if proj is not None:
                        xs.append(proj.project(label, x))

                if xs:
                    points[node] = np.mean(xs, axis=0)

        return points