'''
author: Abolfazl

Estimating the firing rate in two different method.
-  Finding the optimum number of bins 
-  Finding optimum bandwidth for gaussian kernel density estimation

Reference: 
    - Kernel bandwidth optimization in spike rate estimation
    Hideaki Shimazaki & Shigeru Shinomoto 

    - Kernel Density Estimation
    https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
    
    - Kernel density estimation, bandwidth selection
    https://en.wikipedia.org/wiki/Kernel_density_estimation#Bandwidth_selection
'''

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
import numpy as np
import pylab as pl
from os.path import join
import itng.statistics import sshist, find_optimum_bandwidth


with open(join("data.txt"), "r") as f:
    lines = f.readlines()
    spike_times = []
    for line in lines:
        line = [float(i) for i in line.split()]
        spike_times.extend(line)

spike_times = np.asarray(spike_times)

fig, ax = pl.subplots(1, figsize=(6, 4))
ax.set_xlabel('spike times (s)')
ax.set_ylabel("density")


def optimal_num_bins(spike_times, plot=False, ax=None):

    spike_times = np.asarray(spike_times)
    a = kde.sshist(spike_times)
    # print(type(a))
    # print(a[0], a[1], len(a[2]), len(a[3]))
    bins = a[2]
    if plot:
        if ax is None:
            fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(10, 4))
        ax.hist(spike_times, bins=bins, alpha=0.5, density=True)
        # ax.plot(a[3], "k.")

    return bins


print("The optimum number of bins : ",
      optimal_num_bins(spike_times, plot=True, ax=ax).shape)


# Kernel Density Estimation
# Selecting the bandwidth via cross-validation


def find_optimum_bandwidth(spike_times,
                           bandwidths=10 ** np.linspace(-1, 1, 100)):

    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut())
    grid.fit(spike_times[:, None])
    bandwidth = grid.best_params_
    return bandwidth['bandwidth']

bandwidth = find_optimum_bandwidth(spike_times)
print(bandwidth)
# bandwidth = 0.126



# the spikes need to be sorted 
spike_times = np.sort(spike_times)

# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde.fit(spike_times[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(spike_times[:, None])

# ax.fill_between(spike_times, np.exp(logprob), alpha=0.5)
ax.plot(spike_times, np.exp(logprob), alpha=1, lw=2, color="k")
pl.savefig("images/fig.png")
pl.close()
