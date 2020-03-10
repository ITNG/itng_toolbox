import numpy as np
from scipy import fftpack
from scipy import fftpack
from scipy.fftpack import fft, fftshift
from scipy.stats.stats import pearsonr
from scipy.signal import (welch, filtfilt, butter, hilbert)

def fft_1d_real(signal, fs):
    """
    fft from 1 dimensional real signal

    :param signal: [np.array] real signal
    :param fs: [float] frequency sampling in Hz
    :return: [np.array, np.array] frequency, normalized amplitude

    -  example:

    >>> B = 30.0  # max freqeuency to be measured.
    >>> fs = 2 * B
    >>> delta_f = 0.01
    >>> N = int(fs / delta_f)
    >>> T = N / fs
    >>> t = np.linspace(0, T, N)
    >>> nu0, nu1 = 1.5, 22.1
    >>> amp0, amp1, ampNoise = 3.0, 1.0, 1.0
    >>> signal = amp0 * np.sin(2 * np.pi * t * nu0) + amp1 * np.sin(2 * np.pi * t * nu1) +
            ampNoise * np.random.randn(*np.shape(t))
    >>> freq, amp = fft_1d_real(signal, fs)
    >>> pl.plot(freq, amp, lw=2)
    >>> pl.show()

    """

    N = len(signal)
    F = fftpack.fft(signal)
    f = fftpack.fftfreq(N, 1.0 / fs)
    mask = np.where(f >= 0)

    freq = f[mask]
    amplitude = 2.0 * np.abs(F[mask] / N)

    return freq, amplitude


def filter_butter_bandpass(signal, fs, lowcut, highcut, order=5):

    """
    Butterworth filtering function

    :param signal: [np.array] Time series to be filtered
    :param fs: [float] Frequency sampling in Hz
    :param lowcut: [float] Lower value for frequency to be passed in Hz
    :param highcut: [float] Higher value for frequency to be passed in Hz
    :param order: [int] The order of the filter.
    :return: [np.array] filtered frequncy 
    """

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    return filtfilt(b, a, signal)



def pearson_correlation_matrix(data, axis=1):

    """
    calculate the pearson correlation matrix 

    :param data: [np.dnarray (n by num_time_step)] Time series of n nodes, each have num_time_step elements
    :param axis: [int] optional, if 1, each row is considered as a time seri. if 0, each column is a time seri.
    :return: [np.ndarray] correlation matrix
    
    - example

    >>> np.random.seed(1)
    >>> data = np.random.rand(3, 5)
    >>> p = pearson_correlation_matrix(data, axis=1)
    >>> print (p)
    >>> #[[ 0.         -0.63668568  0.6188195 ]
    >>> # [-0.63668568  0.         -0.37240144]
    >>> # [ 0.6188195  -0.37240144  0.        ]]

    """

    assert (data.ndim == 2)

    if axis == 1:
        n = data.shape[0]
    else:
        n = data.shape[1]
    corr = np.zeros((n, n))

    if axis==1:
        for i in range(n):
            for j in range(i + 1, n):
                corr[i, j] = corr[j, i] = pearsonr(data[i, :], data[j, :])[0]
    else:
        for i in range(n):
            for j in range(i + 1, n):
                corr[i, j] = corr[j, i] = pearsonr(data[:, i], data[:, j])[0]
    
    


    return corr
# ------------------------------------------------------------------#

def kuramoto_correlation_matrix(x):

    """
    claculate the Kuramoto correlation 

    :param x: [np.array] array of phases
    :return: [np.ndarray] calculated correlation matrix
    """

    n = len(x)
    cor = np.zeros((n, n))
    for i in range(n):
        cor[i, :] = np.cos(x - x[i])

    return cor


def fwhm2sigma(fwhm):
    """
    Convert a FWHM in a Gaussian kernel to a sigma value

    The FWHM is the width of the kernel, at half of the maximum of the height of the Gaussian.
    Thus, for the standard Gaussian above, the maximum height is ~0.4. The width of the kernel at 0.2 (on the Y axis) is the FWHM. As x = -1.175 and 1.175 when y = 0.2, the FWHM is roughly 2.35.

    :param fwhm: [float] fwhm in gaussian kernel
    :return: sigma in gaussian kernel

    see also:
    https://matthew-brett.github.io/teaching/smoothing_intro.html

    """
    return fwhm / np.sqrt(8 * np.log(2))

def sigma2fwhm(sigma):
    """
    Convert a sigma in a Gaussian kernel to a FWHM value

    The FWHM is the width of the kernel, at half of the maximum of the height of the Gaussian.

    :sigma: sigma in gaussian kernel
    :return: fwhm in gaussian kernel

    >>> sigma2fwhm(1)
    2.3548200450309493

    see also 
    https://matthew-brett.github.io/teaching/smoothing_intro.html
    """
    return sigma * np.sqrt(8 * np.log(2))

# -------------------------------------------------------------#


def smooth_gaussian(x_values, y_values, sigma):

    """
    smoothing signal by gaussian kernel.

    :param x_values: [np.array] x values of given signal
    :param y_values: [np.array] y values if given signal
    :param sigma: [float] sigma in gaussian kernel
    :return: [np.array] smoothed signal

    Gaussian distribution in 1D has the form :
    G(x) = 1/(sqrt(2 * pi) sigma) exp(-x^2/(2*sigma^2))


    >>> FWHM = 4
    >>> n_points = 60
    >>> x_vals = np.arange(n_points)
    >>> y_vals = np.random.normal(size=n_points)
    >>> sigma = fwhm2sigma(FWHM)
    >>> smoothed_g = smooth_gaussian(x_vals, y_vals, sigma)
    >>> plt.plot(x_vals, y_vals, lw=2, label='original')
    >>> plt.plot(x_vals, smoothed_g, lw=3, c='r', label='gaussian')
    """

    smoothed_vals = np.zeros(y_values.shape)
    for x_position in x_values:
        kernel = np.exp(-(x_values - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel / sum(kernel)
        smoothed_vals[x_position] = sum(y_values * kernel)

    return smoothed_vals
