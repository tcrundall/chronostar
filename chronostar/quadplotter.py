from __future__ import division, print_function

import pdb

import numpy as np
import pickle

import chronostar.groupfitter as gf
import chronostar.analyser as an
import matplotlib.pyplot as plt

def plot_quadplots(infile, bayes_fits, init_conditions=None, dir=''):
    """
    Generates many quad plots in the provided directory

    Parameters
    ----------
    infile : str
        The traceback file of a set of stars
    bayes_fits : ???
        Some means of conveying the bayesian fit to each age step
    """

    stars, times, xyzuvw, xyzuvw_cov = pickle.load(open(infile, 'r'))
    nstars = len(xyzuvw)

    # Gather data of spread fits
    naive_spreads = an.get_naive_spreads(xyzuvw)
    assert(len(times) == len(naive_spreads))
    bayes_spreads = gf.get_bayes_spreads(infile)
    assert(len(times) == len(bayes_spreads))

    # Plot spread fits
    plt.clf()
    #f, (ax1, ax2) = plt.subplots(1,2)
    f, ax = plt.subplots(1,1)
    #f.set_size_inches(20,10)
    f.set_size_inches(10,10)

    ax.plot(times, naive_spreads, label="Naive fit")
    ax.plot(times, bayes_spreads, label="Bayes fit")
    ax.set_xlim(times[0],times[-1])
    ax.set_ylim(
        bottom=0.0, top=max(np.max(naive_spreads), np.max(bayes_spreads))
    )
    if init_conditions is not None:
        init_age = init_conditions[13]
        ax.axvline(
            init_age, ax.get_ylim()[0], ax.get_ylim()[1], color='r', ls='--'
        )
    ax.legend(loc=1)

    f.savefig("temp_plot.png")

    return 0
