# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ================================================================ #
def _mesh_arrays(mesh):
    tris = getattr(mesh, "cells", getattr(mesh, "cells"))
    pointsc = getattr(mesh, "cell_centers", getattr(mesh, "pointsc", None))
    pointsf = getattr(mesh, "face_centers", getattr(mesh, "pointsf", None))
    return (
        mesh.points[:, 0],
        mesh.points[:, 1],
        tris,
        pointsc,
        pointsf,
    )


# ================================================================ #
def _set_equal_axes(ax):
    if ax == plt:
        plt.gca().set_aspect(aspect="equal")
        ax.xlabel(r"x")
        ax.ylabel(r"y")
    else:
        ax.set_aspect(aspect="equal")
        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")


# ================================================================ #
def plotmesh(mesh, **kwargs):
    if mesh.dimension != 2:
        raise NotImplementedError("plotmesh currently only supports 2D mesh.")

    ax = kwargs.pop("ax", plt)
    title = kwargs.pop("title", "Mesh")
    alpha = kwargs.pop("alpha", 1)

    x = kwargs.pop("x")
    y = kwargs.pop("y")
    tris = kwargs.pop("tris")

    ax.triplot(x, y, tris, color="k", alpha=alpha)

    _set_equal_axes(ax)

    try:
        ax.set_title(title)
    except Exception:
        ax.title(title)


# ================================================================ #
def plotMeshWithPointData(ax, pdn, pd, x, y, tris, alpha):
    if not isinstance(pd, np.ndarray):
        raise ValueError(f"Problem in data {type(pd)=}")

    if x.shape != pd.shape:
        raise ValueError(f"Problem in data {x.shape=} {pd.shape=}")

    ax.triplot(x, y, tris, color="gray", lw=1, alpha=alpha)

    cnt = ax.tricontourf(x, y, tris, pd, levels=16, cmap="jet")

    clb = plt.colorbar(cnt, ax=ax, shrink=0.6)
    clb.ax.set_title(pdn)

    try:
        ax.set_title(pdn)
    except Exception:
        ax.title(pdn)


# ================================================================ #
def plotMeshWithCellData(ax, cdn, cd, x, y, tris, alpha):
    if tris.shape[0] != cd.shape[0]:
        raise ValueError(
            f"wrong length in '{cdn}' {tris.shape[0]} != {cd.shape[0]}"
        )

    ax.triplot(x, y, tris, color="gray", lw=1, alpha=alpha)

    cnt = ax.tripcolor(
        x,
        y,
        tris,
        facecolors=cd,
        edgecolors="k",
        cmap="jet",
    )

    clb = plt.colorbar(cnt, ax=ax, shrink=0.6)
    clb.ax.set_title(cdn)

    try:
        ax.set_title(cdn)
    except Exception:
        ax.title(cdn)


# ================================================================ #
def meshWithData(mesh, **kwargs):
    if mesh.dimension != 2:
        raise NotImplementedError("meshWithData currently only supports 2D.")

    x, y, tris, pointsc, _ = _mesh_arrays(mesh)

    xc = pointsc[:, 0]
    yc = pointsc[:, 1]

    addplots = kwargs.pop("addplots", [])
    alpha = kwargs.pop("alpha", 0.6)

    if "data" in kwargs:
        point_data = kwargs["data"].get("point", {})
        cell_data = kwargs["data"].get("cell", {})
    else:
        point_data = {}
        cell_data = {}

    if "point_data" in kwargs:
        assert isinstance(kwargs["point_data"], dict)
        point_data.update(kwargs["point_data"])

    if "cell_data" in kwargs:
        assert isinstance(kwargs["cell_data"], dict)
        cell_data.update(kwargs["cell_data"])

    quiver_data = kwargs.get("quiver_data", {})

    nplots = (
        len(point_data)
        + len(cell_data)
        + len(quiver_data)
        + len(addplots)
    )

    if nplots == 0:
        raise ValueError("meshWithData(): no data")

    if "outer" in kwargs:
        import matplotlib.gridspec as gridspec

        inner = gridspec.GridSpecFromSubplotSpec(
            nplots,
            1,
            subplot_spec=kwargs["outer"],
            wspace=0.1,
            hspace=0.1,
        )

        if "fig" not in kwargs:
            raise KeyError("needs argument 'fig'")

        fig = kwargs["fig"]

    else:
        ncols = min(nplots, 3)
        nrows = nplots // 3 + bool(nplots % 3)

        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 4.5, nrows * 4),
            squeeze=False,
        )

    count = 0

    # ------------------------------------------------------------ #
    for pdn, pd in point_data.items():
        if "outer" in kwargs:
            ax = plt.Subplot(fig, inner[count])
        else:
            ax = axs[count // ncols, count % ncols]

        plotMeshWithPointData(ax, pdn, pd, x, y, tris, alpha)

        _set_equal_axes(ax)

        if "title" in kwargs:
            ax.set_title(kwargs["title"])

        fig.add_subplot(ax)

        count += 1

    # ------------------------------------------------------------ #
    for cdn, cd in cell_data.items():
        if "outer" in kwargs:
            ax = plt.Subplot(fig, inner[count])
        else:
            ax = axs[count // ncols, count % ncols]

        plotMeshWithCellData(ax, cdn, cd, x, y, tris, alpha)

        _set_equal_axes(ax)

        fig.add_subplot(ax)

        count += 1

    # ------------------------------------------------------------ #
    for qdn, qd in quiver_data.items():
        if "outer" in kwargs:
            ax = plt.Subplot(fig, inner[count])
        else:
            ax = axs[count // ncols, count % ncols]

        if len(qd) != 2:
            raise ValueError(f"{len(qd)=} {quiver_data=}")

        if qd[0].shape[0] == x.shape[0]:
            ax.quiver(x, y, qd[0], qd[1], units="xy")
        else:
            ax.quiver(xc, yc, qd[0], qd[1], units="xy")

        _set_equal_axes(ax)

        fig.add_subplot(ax)

        count += 1

    # ------------------------------------------------------------ #
    for addplot in addplots:
        if "outer" in kwargs:
            ax = plt.Subplot(fig, inner[count])
        else:
            ax = axs[count // ncols, count % ncols]

        addplot(ax)

        count += 1

    return fig


# ================================================================ #
def meshWithBoundaries(mesh, **kwargs):
    if mesh.dimension != 2:
        raise NotImplementedError("meshWithBoundaries currently only supports 2D.")

    fig = None

    if "outer" in kwargs:
        import matplotlib.gridspec as gridspec

        inner = gridspec.GridSpecFromSubplotSpec(
            1,
            1,
            subplot_spec=kwargs["outer"],
            wspace=0.1,
            hspace=0.1,
        )

        if "fig" not in kwargs:
            raise KeyError("needs argument 'fig'")

        fig = kwargs["fig"]

        ax = plt.Subplot(fig, inner[0])

    elif "ax" in kwargs:
        ax = kwargs.pop("ax")

    else:
        ax = plt

    lines = mesh.faces
    bdrylabels = mesh.bdrylabels

    x, y, tris, _, _ = _mesh_arrays(mesh)

    ax.triplot(x, y, tris, color="k")

    _set_equal_axes(ax)

    pltcolors = "bgrcmykbgrcmyk"

    patches = []

    for i, (color, edges) in enumerate(bdrylabels.items()):
        col = pltcolors[i % len(pltcolors)]

        patches.append(mpatches.Patch(color=col, label=f"{color}"))

        for ie in edges:
            ax.plot(x[lines[ie]], y[lines[ie]], color=col, lw=4)

    # ------------------------------------------------------------ #
    if "celllabels" in kwargs:
        celllabels = kwargs.pop("celllabels")

        cnt = ax.tripcolor(
            x,
            y,
            tris,
            facecolors=celllabels,
            edgecolors="k",
            cmap="jet",
            alpha=0.4,
        )

        clb = plt.colorbar(cnt)
        clb.set_label("cellcolors")

    # ------------------------------------------------------------ #
    if "cellsoflabel" in kwargs:
        cellsoflabel = kwargs.pop("cellsoflabel")

        celllabels = np.empty(tris.shape[0])

        for color, cells in cellsoflabel.items():
            celllabels[cells] = color

        ax.tripcolor(
            x,
            y,
            tris,
            facecolors=celllabels,
            edgecolors="k",
            cmap="jet",
            alpha=0.4,
        )

    ax.legend(handles=patches)

    title = "Mesh and Boundary Labels"

    try:
        ax.set_title(title)
    except Exception:
        ax.title(title)

    if fig:
        fig.add_subplot(ax)

    return fig