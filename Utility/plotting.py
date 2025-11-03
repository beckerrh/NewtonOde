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
    for k, v in y.items():
        kwp = kwargs_plot[k] if k in kwargs_plot else {}
        count += 1 if v.ndim == 1 else v.shape[1]
        if step:
            ax.step(x[:-1], v, where='post', label=k, **kwp)
        else:
            if scale == "loglog":
                ax.loglog(x, v, label=k, **kwp)
            else:
                ax.plot(x, v, label=k, **kwp)
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
        ax.set_title(k)
        plot_solutions_single(ax, v)
    # plt.show()
def plot_error_curves(plot_dicts):
    for k,v in list(plot_dicts.items()):
        ns = v['x']
        u_plot = v['y']
        v['scale'] = 'loglog'
        plot_dicts[k]['kwargs'] = {}
        for k2, errs in list(u_plot.items()):
            a,b = np.polyfit(np.log(ns), np.log(errs), 1)
            kslope = k2+f" {np.fabs(a):6.2f}"
            plot_dicts[k]['y'][kslope] = np.exp(b)*ns**a
            plot_dicts[k]['kwargs'][k2] = {'ls': '--', 'marker': 'o'}
            plot_dicts[k]['kwargs'][kslope] = {'ls': ':', 'color': 'k'}
    plot_solutions(plot_dicts)

