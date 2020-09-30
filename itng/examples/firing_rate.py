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
from itng.statistics import (sshist, optimal_bandwidth,
                             optimal_num_bins)


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


bins = optimal_num_bins(spike_times) 
print("The optimum number of bins : ", len(bins))


# Kernel Density Estimation
# Selecting the bandwidth via cross-validation

# bandwidth = optimal_bandwidth(spike_times)
# print(bandwidth)
bandwidth = 0.126

# the spikes need to be sorted
spike_times = np.sort(spike_times)

# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde.fit(spike_times[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(spike_times[:, None])

ax.fill_between(spike_times, np.exp(logprob), alpha=0.5)
ax.plot(spike_times, np.exp(logprob), alpha=1, lw=2, color="k")
pl.savefig("images/fig.png")
pl.close()
