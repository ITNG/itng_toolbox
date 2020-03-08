import os
# import bct
# import igraph
import numpy as np
import pylab as pl
from copy import copy
import networkx as nx
from random import shuffle
import matplotlib.pyplot as plt


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
