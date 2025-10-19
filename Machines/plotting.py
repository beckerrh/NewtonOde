import matplotlib.pyplot as plt
import numpy as np

#==================================================================
# Commençons par la fonction de visu
def plot_solutions_single(ax, kwargs):
    t_plot = kwargs.pop('t_plot')
    u_plot = kwargs.pop('u_plot')
    for k, v in u_plot.items():
        ax.plot(t_plot, v, label=k)
    ax.legend()
    ax.set_xlabel("t")
    ax.grid(True)
def plot_solutions(plot_dicts):
    for i, (k,v) in enumerate(plot_dicts.items()):
        ax = plt.subplot(len(plot_dicts), 1, i+1)
        ax.set_title(k)
        plot_solutions_single(ax, v)
    plt.show()

