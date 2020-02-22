from scipy.signal import welch, filtfilt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import butter, hilbert
import networkx as nx
from time import time
import numpy as np
import pylab as pl
import igraph
import os


# ---------------------------------------------------------------#


def isi(spikeTimes, gids):
    """
    calculate interspike interval of given spike train

    :param spikeTimes: time of spikes 1 dimensional array, list or tuple
    :param gids: global ID of neurons
    :return: inter spike interval 1d numpy array
    """

    neurons = np.unique(gids)
    t_pop = []
    for i in neurons:
        indices = np.where(gids == i)
        spikes = np.sort(spikeTimes[indices])
        isi_i = np.diff(spikes)
        t_pop.extend(isi_i)

    return np.asarray(t_pop)
# ---------------------------------------------------------------#


def spike_synchrony(ts, gids, threshold_num_spikes=10):
    """
    calculate spike synchrony. 

    :param ts: time of spikes 1 dimensional array, list or tuple.
    :param gids: global ID of neurons.
    :param threshold_num_spikes: minimum number of spikes required for calculation of measure.
    :return: [float] spike synchrony in [0, 1].

    Reference
    
    -  Tiesinga, P. H. & Sejnowski, T. J. Rapid temporal modulation of synchrony by competition in cortical interneuron networks. Neural computation 16, 251–275 (2004).
    """

    if len(ts) < threshold_num_spikes:
        return 0.0
    else:
        num_neurons = len(np.unique(gids))
        t = isi(ts, gids)
        t2 = t*t
        t_m = np.mean(t)
        t2_m = np.mean(t2)
        t_m2 = t_m*t_m
        sync = ((np.sqrt(t2_m - t_m2) /
                 (t_m+0.0))-1.0)/(np.sqrt(num_neurons)+0.0)

        return sync
# ---------------------------------------------------------------#

def voltage_synchrony(self, voltages):
    """
    calculate voltage synchrony.

    :param voltages: [ndarray, nested list or nested tuple (number of nodes by number of time steps)] votages of n nodes. 
    :return: [float] voltage synchrony in [0, 1]

    Reference:

    -  Lim, W. & Kim, S. Y. Coupling-induced spiking coherence in coupled subthreshold neurons. Int. J. Mod. Phys. B 23, 2149–2157 (2009)

    """


def calculate_NMI(comm1, comm2, method="nmi"):
    """
    Compares two community structures

    :param comm1: the first community structure as a membership list or as a Clustering object.
    :param comm2: the second community structure as a membership list or as a Clustering object.
    :param method: [string] defaults to ["nmi"] the measure to use. "vi" or "meila" means the variation of information metric of Meila (2003), "nmi" or "danon" means the normalized mutual information as defined by Danon et al (2005), "split-join" means the split-join distance of van Dongen (2000), "rand" means the Rand index of Rand (1971), "adjusted_rand" means the adjusted Rand index of Hubert and Arabie (1985).
    :return: [float] the calculated measure.

    Reference: 

    -  Meila M: Comparing clusterings by the variation of information. In: Scholkopf B, Warmuth MK (eds). Learning Theory and Kernel Machines: 16th Annual Conference on Computational Learning Theory and 7th Kernel Workship, COLT/Kernel 2003, Washington, DC, USA. Lecture Notes in Computer Science, vol. 2777, Springer, 2003. ISBN: 978-3-540-40720-1.

    -  Danon L, Diaz-Guilera A, Duch J, Arenas A: Comparing community structure identification. J Stat Mech P09008, 2005.

    -  van Dongen D: Performance criteria for graph clustering and Markov cluster experiments. Technical Report INS-R0012, National Research Institute for Mathematics and Computer Science in the Netherlands, Amsterdam, May 2000.

    -  Rand WM: Objective criteria for the evaluation of clustering methods. J Am Stat Assoc 66(336):846-850, 1971.

    -  Hubert L and Arabie P: Comparing partitions. Journal of Classification 2:193-218, 1985.
    """

    nmi = igraph.compare_communities(
        communities1, comm2, method='nmi', remove_none=False)
    return nmi


# if __name__ == "__main__":
#     x = MyClass()
