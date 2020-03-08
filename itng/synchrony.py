
def spike_synchrony(self, ts, gids, threshold_num_spikes=10):
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
