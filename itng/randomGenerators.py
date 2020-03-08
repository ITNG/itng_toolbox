import numpy as np
from numpy import power
# from powerlaw import plot_pdf, Fit, pdf
# from networkx.utils import powerlaw_sequence


def power_law(self, N, e, xmin, xmax):
    '''
    generate a power law distribution of integers from uniform distribution 

    :param N: [int] number of data in powerlaw distribution (pwd).
    :param e: [int, float] exponent of the pwd.
    :param xmin: [int] min value in pwd.
    :param xmax: [int] max value in pwd.
    :return: [numpy array of int] the power law distribution

    becuse the numbers will use as degree of nodes, the sum of degrees should be an even number.

    Reference:

    - http://mathworld.wolfram.com/RandomNumber.html
    '''

    from numpy.random import rand, randint

    data = np.zeros(N, dtype=int)
    x1p = power(xmax, (e+1.0))
    x0p = power(xmin, (e+1.0))
    alpha = 1.0/(e+1.0)

    r = rand(N)

    for i in range(N):
        r = rand()
        data[i] = int(np.round(power(((x1p - x0p)*r + x0p), alpha)))

    # sum of velues should be positive
    # else choose a random node and add one

    if ((np.sum(data) % 2) != 0):
        i = randint(0, N)
        data[i] = data[i]+1

    return data
#--------------------------------------------------------------#
