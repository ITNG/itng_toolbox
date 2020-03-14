import os
# import bct
# import igraph
import numpy as np
import pylab as pl
from copy import copy
import networkx as nx
from random import shuffle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Drawing(object):

    """
    draw graphs in different representation.
    """

    def __init__(self):
        pass

    @staticmethod
    def plot_adjacency_matrix(G,
                              file_name='R.png',
                              cmap='afmhot',
                              figsize=(5, 5),
                              labelsize=None,
                              xticks=True,
                              yticks=True,
                              vmax=None,
                              vmin=None,
                              colorbar=True,
                              ax=None):
        """
        plot the adjacency matrix of given graph

        :param G: networkx graph
        :param file_name: [string] file name to save the figure. 
        :param figsize: (float, float) the dimension of the figure.

        """

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        saveFig = False

        if ax is None:
            fig, ax = pl.subplots(1, figsize=figsize)
            saveFig = True

        adj = nx.to_numpy_array(G)

        im = ax.imshow(adj, interpolation='nearest', cmap=cmap,
                       vmax=vmax, vmin=vmin, origin="lower")

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = pl.colorbar(im, cax=cax)

        if labelsize:
            cbar.ax.tick_params(labelsize=labelsize)

        if xticks == False:
            ax.set_xticks([])

        if yticks == False:
            ax.set_yticks([])

        if saveFig:
            pl.savefig(file_name)
            pl.close()
# ------------------------------------------------------------------#


def make_grid(nrows,
              ncols,
              left=0.05,
              right=0.95,
              bottom=0.05,
              top=0.95,
              hspace=0.2,
              wspace=0.2):
    """
    make grided figure

    :param nrows: [int] number of rows in figure
    :param ncols: [int] number of columns in figure
    :param left, right, top, bottom: [float], optional. Extent of the subplots as a fraction of figure width or height. Left cannot be larger than right, and bottom cannot be larger than top.
    :param hspace: [flaot], optional. The amount of height reserved for space between subplots, expressed as a fraction of the average axis height.
    :param wspace: [float], optional. The amount of width reserved for space between subplots, expressed as a fraction of the average axis width.
    :return: axes of gridded figure

    >>> import pylab as pl
    >>> fig = pl.figure(figsize=(20, 15))
    >>> ax = make_grid(2, 3, 0.1, 0.95, 0.1, 0.95, 0.2, 0.2)
    >>> ax[0][0].plot(range(10), marker="o")
    >>> pl.show()

    """

    gs = GridSpec(nrows, ncols)
    gs.update(left=left, right=right,
              hspace=hspace, wspace=wspace,
              bottom=bottom, top=top)
    ax = []
    for i in range(nrows):
        ax_row = []
        for j in range(ncols):
            ax_row.append(pl.subplot(gs[i, j]))
        ax.append(ax_row)
        
    return ax
# ---------------------------------------------------------------------- #


def plot_scatter(x, y,
                 ax,
                 xlabel=None,
                 ylabel=None,
                 xlim=None,
                 ylim=None,
                 color="k",
                 alpha=0.4,
                 markersize=10,
                 labelsize=14,
                 title=None):
    """
    scatter plot.

    :param x: [np.array] values on x axis.
    :param y: [np.array] values on y axis, with save size of x values.
    :param ax: axis of figure to plot the figure.
    :param xlabel: if given, label of x axis.
    :param ylabel: if given, label of y axis.
    :param xlim: [float, float] limit of x values.
    :param ylim: [float, float] limit of y values.
    :param color: [string] color of markers.
    :param alpha: [float] opacity of markers in [0, 1].
    :param markersize: [int] size of marker.
    :param title: title of fiure.
    :return: axis with plot
    """

    xl = x.reshape(-1)
    yl = y.reshape(-1)

    assert (len(xl) == len(yl))

    ax.scatter(xl, yl, s=markersize,
               color=color, alpha=alpha)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)

    ax.tick_params(labelsize=labelsize)

    if (xlim is None) and (ylim is None):
        ax.margins(x=0.02, y=0.02)

    if title:
        ax.set_title(title)

    return ax
# ---------------------------------------------------------------------- #


def fit_line(X,
             Y,
             ax,
             color="k",
             alpha=0.4,
             labelsize=14,
             ):
            
    """
    fit a line to given values.

    :param X: [np.array] values on x axis
    :param Y: [np.array] values on y axis
    :param ax: axis to plot the line
    :param color: [string] color of the line
    :param alpha: [float] opacity of the line
    :param labelsize: size of labels on axises
    :return: axis with plot
    """

    import statsmodels.formula.api as smf

    x = X.reshape(-1)
    y = Y.reshape(-1)

    assert(len(x) == len(y))

    df_data = pd.DataFrame({"y": y, "x": x})
    model = smf.ols("y ~ 1 + x", df_data)
    result = model.fit()
    ax.plot(x, result.fittedvalues, lw=2, c="r",
            label="y=ax+b, a=%.3f" % result.params["x"])
    ax.legend(loc="lower right")

    return ax

# ---------------------------------------------------------------------- #
