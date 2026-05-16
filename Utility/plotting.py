import matplotlib.pyplot as plt
import numpy as np


#==================================================================
def plot_mesh(ax, mesh, y=0.0, label="mesh"):
    ax.plot(mesh, y*np.ones_like(mesh), "|", markersize=12, label=label)
    ax.set_yticks([])
    ax.grid(True)
    ax.legend()
#==================================================================
def plot_solutions_single(ax, data):
    data = data.copy()

    x = data.pop('x')
    y = data.pop('y')
    xlabel = data.pop('xlabel', None)
    ylabel = data.pop('ylabel', None)
    step = data.pop('type', 'p1') == 'step'
    scale = data.pop('scale', 'normal')
    if scale not in ["normal", "loglog"]:
        raise ValueError(f"unknown scale: {scale}")
    if step and scale == "loglog":
        raise ValueError("step plot with loglog scale is not implemented")
    kwargs_plot = data.pop('kwargs', {})
    no_legend = data.pop('no_legend', False)

    single_x = not isinstance(x, list)
    if not single_x:
        assert len(x) == len(y), f"{len(x)} != {len(y)}"
    for count, (k, v) in enumerate(y.items()):
        xp = x if single_x else x[count]
        kwp = kwargs_plot[k] if k in kwargs_plot else {}
        if step:
            ax.step(xp[:-1], v, where='post', label=k, **kwp)
        else:
            if scale == "loglog":
                ax.loglog(xp, v, label=k, **kwp)
            else:
                ax.plot(xp, v, label=k, **kwp)
    if not no_legend: ax.legend()
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(True)
    if data:
        raise KeyError(f"unused plot arguments: {data}")
#==================================================================
def plot_solutions(plot_dicts, title=None):
    n_plots = len(plot_dicts)
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(8, 8))
    if title: fig.suptitle(title)
    if not isinstance(axes, np.ndarray): axes = [axes]
    for ax, (k,v) in zip(axes, plot_dicts.items()):
        # print(f"plot_solutions {k=}")
        ax.set_title(k)
        try:
            plot_solutions_single(ax, v)
        except Exception as e:
            raise KeyError(f"problem in key {k}") from e
    fig.tight_layout(rect=[0, 0, 1, 0.95])

#==================================================================
def plot_error_curves(plot_dicts):
    import copy
    plot_dicts = copy.deepcopy(plot_dicts)
    for k,v in list(plot_dicts.items()):
        if 'kwargs' in plot_dicts[k]: raise KeyError(f"key {k} cannot have kwargs")
        ns = v['x']
        count = 0
        single_ns = not isinstance(ns, list)
        u_plot = v['y']
        if single_ns:
            nsp = ns
        else: assert len(ns) == len(u_plot), f"{len(ns)} != {len(u_plot)} in {k=}"
        plot_dicts[k]['kwargs'] = {}
        for k2, errs in list(u_plot.items()):
            if not single_ns:
                nsp = ns[count]
            count += 1
            try:
                a,b = np.polyfit(np.log(nsp), np.log(errs), 1)
            except Exception as e:
                raise KeyError(f"problem in key {k2} {len(errs)=} {len(nsp)=}") from e
            kslope = k2+f" {np.fabs(a):6.2f}"
            plot_dicts[k]['y'][kslope] = np.exp(b)*nsp**a
            plot_dicts[k]['kwargs'][k2] = {'ls': '--', 'marker': 'o'}
            plot_dicts[k]['kwargs'][kslope] = {'ls': ':', 'color': 'k'}
    # print(f"{plot_dicts=}")
    for k in plot_dicts.keys():
        plot_dicts[k]['scale'] = 'loglog'
        if isinstance(plot_dicts[k]['x'], list):
            dl = [x  for _ in range(2) for x in plot_dicts[k]['x']]
            plot_dicts[k]['x'] = dl

    plot_solutions(plot_dicts)

#==================================================================
def add_mesh1d(plot_dict, mesh):
    plot_dict['Mesh'] = {}
    plot_dict["Mesh"]['x'] = mesh
    hinv = 1.0 / (mesh[1:] - mesh[:-1])
    h0 = np.min(hinv)
    hinv /= h0
    plot_dict["Mesh"]['y'] = {rf'$\frac{{{1/h0:.2e}}}{{h}}$': hinv}
    plot_dict["Mesh"]['type'] = 'step'
    cand_cell = ['zeta', 'eta', 'mu', 'err']
    cell_dict={}
    return plot_dict
