import matplotlib.pyplot as plt
import numpy as np

#==================================================================
# Commençons par la fonction de visu
def plot_solutions_single(ax, kwargs):
    # print(f"{kwargs=}")
    x, y = kwargs.pop('x'), kwargs.pop('y')
    xlabel, ylabel = kwargs.pop('xlabel',None), kwargs.pop('ylabel',None)
    step = kwargs.pop('type', 'p1')=='step'
    assert isinstance(y, dict)
    scale = kwargs.pop('scale', 'normal')
    kwargs_plot = kwargs.pop('kwargs', {})
    count = 0
    single_x = not isinstance(x, list)
    if single_x: xp = x
    else: assert len(x) == len(y), f"{len(x)} != {len(y)}"
    for k, v in y.items():
        # print(f"plot_solutions_single {k=}")
        if not single_x: xp = x[count]
        kwp = kwargs_plot[k] if k in kwargs_plot else {}
        count += 1 if v.ndim == 1 else v.shape[1]
        if step:
            ax.step(xp[:-1], v, where='post', label=k, **kwp)
        else:
            if scale == "loglog":
                ax.loglog(xp, v, label=k, **kwp)
            else:
                ax.plot(xp, v, label=k, **kwp)
    if not 'no_legend' in kwargs: ax.legend()
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_xlabel(ylabel)
    ax.grid(True)
def plot_solutions(plot_dicts, title=None):
    n_plots = len(plot_dicts)
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(8, 8))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if title: fig.suptitle(title)
    if not isinstance(axes, np.ndarray): axes = [axes]
    for ax, (k,v) in zip(axes, plot_dicts.items()):
        # print(f"plot_solutions {k=}")
        ax.set_title(k)
        try:
            plot_solutions_single(ax, v)
        except:
            raise KeyError(f"problem in key {k}")
    for k,v in plot_dicts.items():
        if len(v):
            print(f"unused arguments {k}:{v}")
    # plt.show()
def plot_error_curves(plot_dicts):
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
            except:
                raise KeyError(f"problem in key {k2} {len(errs)=} {len(nsp)=}")
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

