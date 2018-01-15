from __future__ import division, print_function

import pdb

import numpy as np
import pickle

import chronostar.groupfitter as gf
import chronostar.analyser as an
import chronostar.error_ellipse as ee
import matplotlib.pyplot as plt

def plot_quadplots(infile, bayes_spreads=None, naive_spreads=None, time_probs=None,
                   init_conditions=None, dir=''):
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
    if naive_spreads is None:
        naive_spreads = an.get_naive_spreads(xyzuvw)
    if bayes_spreads is None or time_probs is None:
        bayes_spreads, time_probs = gf.get_bayes_spreads(infile)
    assert(len(times) == len(naive_spreads))
    assert(len(times) == len(bayes_spreads))
    pdb.set_trace() # HARD COPY TIME_PROBS INTO quadplotter_demo.py

    # Plot spread fits
    plt.clf()
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    f.set_size_inches(50, 50)

    # Plotting XY traceback of each star
    dim1 = 0 # for X
    dim2 = 1 # for Y

    cov_ix1 = [[dim1, dim2], [dim1, dim2]]
    cov_ix2 = [[dim1, dim1], [dim2, dim2]]
    for i in range(nstars):
        ax1.plot(xyzuvw[i, :, dim1], xyzuvw[i, :, dim2], 'b-')

        cov_start = xyzuvw_cov[i,  0, cov_ix1, cov_ix2]
        cov_end = xyzuvw_cov[i, -1, cov_ix1, cov_ix2]
        ee.plot_cov_ellipse(
            cov_start,
            [xyzuvw[i, 0, dim1], xyzuvw[i, 0, dim2]],
            color='g', alpha=0.1, ax=ax1
        )
        ee.plot_cov_ellipse(
            cov_end,
            [xyzuvw[i, -1, dim1], xyzuvw[i, -1, dim2]],
            color='r', alpha=0.1, ax=ax1
        )

    POS_RANGE = 300
    ax1.set(aspect='equal')
    ax1.set_xlabel('X [pc]')
    ax1.set_ylabel('Y [pc]')
    ax1.set_xlim(POS_RANGE, -POS_RANGE) # note inverse X axis
    ax1.set_ylim(-POS_RANGE, POS_RANGE)

    ax2.plot(times, naive_spreads, label="Naive fit")
    ax2.plot(times, bayes_spreads, label="Bayes fit")
    ax2.set_xlim(times[0],times[-1])
    ax2.set_ylim(
        bottom=0.0, top=max(np.max(naive_spreads), np.max(bayes_spreads))
    )
    if init_conditions is not None:
        init_age = init_conditions[13]
        ax2.axvline(
            init_age, ax2.get_ylim()[0], ax2.get_ylim()[1], color='r', ls='--'
        )
    ax2.legend(loc=1)
    ax2.set_xlabel("Traceback Time [Myr]")
    ax2.set_ylabel("Radius of average spread in XYZ [pc]")

    # plot UV traceback
    dim1 = 3  # for U
    dim2 = 4  # for V

    cov_ix1 = [[dim1, dim2], [dim1, dim2]]
    cov_ix2 = [[dim1, dim1], [dim2, dim2]]
    for i in range(nstars):
        ax3.plot(xyzuvw[i, :, dim1], xyzuvw[i, :, dim2], 'b-')

        cov_start = xyzuvw_cov[i, 0, cov_ix1, cov_ix2]
        cov_end = xyzuvw_cov[i, -1, cov_ix1, cov_ix2]
        ee.plot_cov_ellipse(
            cov_start,
            [xyzuvw[i, 0, dim1], xyzuvw[i, 0, dim2]],
            color='g', alpha=0.1, ax=ax3
        )
        ee.plot_cov_ellipse(
            cov_end,
            [xyzuvw[i, -1, dim1], xyzuvw[i, -1, dim2]],
            color='r', alpha=0.1, ax=ax3
        )

    VEL_RANGE = 50
    ax3.set(aspect='equal')
    ax3.set_xlabel('U [km/s]')
    ax3.set_ylabel('V [km/s]')
    ax3.set_xlim(VEL_RANGE, -VEL_RANGE)  # note inverse X axis
    ax3.set_ylim(-VEL_RANGE, VEL_RANGE)

    # PLot age PDF
    ax4.plot(times, time_probs)
    ax4.set_xlim(times[0],times[-1])
    ax4.set_ylim(bottom=0.0, top=max(time_probs))
#    if init_conditions is not None:
#        init_age = init_conditions[13]
#        ax4.axvline(
#            init_age, ax4.get_ylim()[0], ax4.get_ylim()[1], color='r', ls='--'
#        )
    ax4.set_xlabel("Traceback Time [Myr]")
    ax4.set_ylabel("Age likelihoods (non-normalised)")

    f.savefig("temp_plot.png")

    return 0
