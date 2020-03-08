
def isi(self, spiketimes, gids):
    """
    calculate interspike interval of given spike train

    :param spikeTimes: time of spikes, 1 dimensional array, list or tuple
    :param gids: global ID of neurons
    :return: inter spike interval 1d numpy array
    """

    neurons = np.unique(gids)
    t_pop = []
    for i in neurons:
        indices = np.where(gids == i)
        spikes = np.sort(spiketimes[indices])
        isi_i = np.diff(spikes)
        t_pop.extend(isi_i)

    return np.asarray(t_pop)
# ---------------------------------------------------------------#


def cv(self, ts, gids):
    """
    Compute the coefficient of variation.

    !todo
    """
    pass
# ---------------------------------------------------------------#


def mean_firing_rate(self, spiketrain, t_start=None, t_stop=None):
    
    """
    Return the firing rate of the spike train.


    :param spiketrain: [np.ndarray] the spike times
    :param t_start: [float] The start time to use for the interval
    :param t_stop: [float] The end time to use for the interval
    :return: The firing rate of the spiketrain

    #!todo
    
    """
    pass
