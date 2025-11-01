import matplotlib.pyplot as plt
import numpy as np

#==================================================================
# Commençons par la fonction de visu
def plot_solutions_single(ax, kwargs):
    # print(f"{kwargs=}")
    x = kwargs.pop('x')
    y = kwargs.pop('y')
    assert isinstance(y, dict)
    scale = kwargs.pop('scale', 'normal')
    kwargs_plot = kwargs.pop('kwargs', {})
    count = 0
    for k, v in y.items():
        kwp = kwargs_plot[k] if k in kwargs_plot else {}
        count += 1 if v.ndim == 1 else v.shape[1]
        if scale == "loglog":
            ax.loglog(x, v, label=k, **kwp)
        else:
            ax.plot(x, v, label=k, **kwp)
    if not 'no_legend' in kwargs: ax.legend()
    ax.set_xlabel("t")
    ax.grid(True)
def plot_solutions(plot_dicts):
    n_plots = len(plot_dicts)
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(8, 8))
    fig.tight_layout()
    if not isinstance(axes, np.ndarray): axes = [axes]
    for ax, (k,v) in zip(axes, plot_dicts.items()):
        ax.set_title(k)
        plot_solutions_single(ax, v)
    plt.show()
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
    # for k,v in plot_dicts.items():
    #     print(f"{k=}")
    #     for k2, v in v['u_plot'].items():
    #         print(f"\t {k2} {v}")
    plot_solutions(plot_dicts)

