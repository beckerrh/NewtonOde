import matplotlib.pyplot as plt
import numpy as np

#==================================================================
# Commençons par la fonction de visu
def plot_solutions_single(ax, kwargs):
    t_plot = kwargs.pop('t_plot')
    u_plot = kwargs.pop('u_plot')
    count = 0
    for k, v in u_plot.items():
        count += 1 if v.ndim == 1 else v.shape[1]
        ax.plot(t_plot, v, label=k)
    if not 'no_legend' in kwargs: ax.legend()
    ax.set_xlabel("t")
    ax.grid(True)
def plot_solutions(plot_dicts):
    n_plots = len(plot_dicts)
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(8, 8))
    fig.tight_layout()
    if not isinstance(axes, np.ndarray): axes = [axes]
    for ax, (k,v) in zip(axes, plot_dicts.items()):
    # for i, (k,v) in enumerate(plot_dicts.items()):
    #     ax = plt.subplot(n_plots, 1, i+1)
        ax.set_title(k)
        plot_solutions_single(ax, v)
    plt.show()

