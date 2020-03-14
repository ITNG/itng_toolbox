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
    plot the adjacency matrix of given graph
    """

    def __init__(self):
        pass

    @staticmethod
    def plot_adjacency_matrix(G,
                              fileName='R.png',
                              cmap='afmhot',
                              figsize=(5, 5),
                              labelsize=None,
                              xticks=True,
                              yticks=True,
                              vmax=None,
                              vmin=None,
                              colorbar=True,
                              ax=None):

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
            pl.savefig(fileName)
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
    :param left, right, top, bottom: float, optional. Extent of the subplots as a fraction of figure width or height. Left cannot be larger than right, and bottom cannot be larger than top.
    :param hspace: The amount of height reserved for space between subplots, expressed as a fraction of the average axis height.
    :param wspace: The amount of width reserved for space between subplots, expressed as a fraction of the average axis width.
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
