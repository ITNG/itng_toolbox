import numpy as np
import pylab as pl
from os.path import join
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV


# ---------------------------------------------------------------#


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
# ---------------------------------------------------------------#


def sshist(x, N=list(range(2, 501)), SN=30):
    """
    Returns the optimal number of bins in a histogram used for density
    estimation.

    Optimization principle is to minimize expected L2 loss function between
    the histogram and an unknown underlying density function.
    An assumption made is merely that samples are drawn from the density
    independently each other.

    The optimal binwidth D* is obtained as a minimizer of the formula,
    (2K-V) / D^2,
    where K and V are mean and variance of sample counts across bins with width
    D. Optimal number of bins is given as (max(x) - min(x)) / D.

    Parameters
    ----------
    x : array_like
        One-dimensional data to fit histogram to.
    N : array_like, optional
        Array containing number of histogram bins to evaluate for fit.
        Default value = 500.
    SN : double, optional
        Scalar natural number defining number of bins for shift-averaging.

    Returns
    -------
    optN : int
        Optimal number of bins to represent the data in X
    N : double
        Maximum number of bins to be evaluated. Default value = 500.
    C : array_like
        Cost function C[i] of evaluating histogram fit with N[i] bins

    See Also
    --------
    sskernel, ssvkernel

    References
    ----------
    .. [1] H. Shimazaki and S. Shinomoto, "A method for selecting the bin size
           of a time histogram," in  Neural Computation 19(6), 1503-1527, 2007
           http://dx.doi.org/10.1162/neco.2007.19.6.1503
    
    .. [2] the code is taken from here and revided to adapt python3 and resolve
           some bugs: https://github.com/cooperlab/AdaptiveKDE
    """

    # determine range of input 'x'
    x_min = np.min(x)
    x_max = np.max(x)

    # get smallest difference 'dx' between all pairwise samples
    buf = np.abs(np.diff(np.sort(x)))
    dx = min(buf[buf > 0])

    # setup bins to evaluate
    N_MIN = 2
    # N_MAX = min(np.floor((x_max - x_min) / (2*dx)), max(N))
    N_MAX = int(min(np.floor((x_max - x_min) / (2*dx)), max(N)))
    N = list(range(N_MIN, N_MAX+1))
    D = (x_max - x_min) / N

    # compute cost function over each possible number of bins
    Cs = np.zeros((len(N), SN))
    for i, n in enumerate(N):  # loop over number of bins
        shift = np.linspace(0, D[i], SN)
        for p, sh in enumerate(shift):  # loop over shift window positions

            # define bin edges
            edges = np.linspace(x_min + sh - D[i]/2,
                                x_max + sh - D[i]/2, N[i]+1)

            # count number of events in these bins
            ki = np.histogram(x, edges)

            # get mean and variance of events
            k = ki[0].mean()
            v = np.sum((ki[0] - k)**2) / N[i]

            Cs[i, p] = (2*k - v) / D[i]**2

    # average over shift window
    C = Cs.mean(axis=1)

    # get bin count that minimizes cost C
    idx = np.argmin(C)
    optN = N[idx]
    optD = D[idx]
    edges = np.linspace(x_min, x_max, optN)

    return optN, optD, edges, C, N
# ---------------------------------------------------------------#


def optimal_num_bins(spike_times, plot=False, ax=None):
    '''
    find optimum number of bins using sshist for estimation of
    firing rate.
    '''
    spike_times = np.asarray(spike_times)
    a = sshist(spike_times)
    # print(type(a))
    # print(a[0], a[1], len(a[2]), len(a[3]))
    bins = a[2]
    if plot:
        if ax is None:
            fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(10, 4))
        ax.hist(spike_times, bins=bins, alpha=0.5, density=True)
        # ax.plot(a[3], "k.")

    return bins
# ---------------------------------------------------------------#


def optimal_bandwidth(spike_times,
                           bandwidths=10 ** np.linspace(-1, 1, 100)):
    """
    find the optimum bandwith for spike rate estimation

    Reference: 
        -  https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
    """
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut())
    grid.fit(spike_times[:, None])
    bandwidth = grid.best_params_
    return bandwidth['bandwidth']
