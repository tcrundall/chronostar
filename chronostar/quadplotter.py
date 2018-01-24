from __future__ import division, print_function

import pdb

import numpy as np
import pickle

import chronostar.groupfitter as gf
import chronostar.analyser as an
import chronostar.error_ellipse as ee
import matplotlib.pyplot as plt

def calc_area(ys):
    """Calculate the area under the curve provided

    Parameters
    ----------
    ys : [n] array

    Output
    ------
    Approximate area under curve
    """
    assert(np.min(ys) >= 0)
    nrecs = len(ys) - 1

    total_area = 0
    for i in range(nrecs):
        total_area += 0.5 * (ys[i] + ys[i+1]) / nrecs

    return total_area

def plot_sub_traceback(xyzuvw, xyzuvw_cov, times, dim1, dim2, ax):
    """Plot the 2D traceback of stars with their error ellipses

    Parameters
    ----------
    xyzuvw : [nstars, ntimes, 6] array
        exact orbital position in XYZ (pc) and UVW (km/s)
    xyzuvw_cov : [nstars, ntimes, 6, 6] array
        covariance matrix for each star at each time step
    times : [ntimes] array
        times corresponding to each traceback step
    dim1, dim2 : ints
        Denotes which phase dimensions will be plotted [0,1,2,3,4,5] -> [XYZUVW]
    ax : pyplot axes object
        the axes to be plotted in
    """
    labels = ['X [pc]', 'Y [pc]', 'Z [pc]', 'U [km/s]', 'V [km/s]', 'W [km/s]']
    # X and U are negative as per convention
    axis_ranges = [-300, 300, 300, -50, 50, 50]

    nstars = len(xyzuvw)

    cov_ix1 = [[dim1, dim2], [dim1, dim2]]
    cov_ix2 = [[dim1, dim1], [dim2, dim2]]
    for i in range(nstars):
        ax.plot(xyzuvw[i, :, dim1], xyzuvw[i, :, dim2], 'b-')

        cov_start = xyzuvw_cov[i,  0, cov_ix1, cov_ix2]
        cov_end = xyzuvw_cov[i, -1, cov_ix1, cov_ix2]
        ee.plot_cov_ellipse(
            cov_start,
            [xyzuvw[i, 0, dim1], xyzuvw[i, 0, dim2]],
            color='g', alpha=0.1, ax=ax
        )
        ee.plot_cov_ellipse(
            cov_end,
            [xyzuvw[i, -1, dim1], xyzuvw[i, -1, dim2]],
            color='r', alpha=0.1, ax=ax
        )

    ax.set(aspect='equal')
    ax.set_xlabel(labels[dim1])
    ax.set_ylabel(labels[dim2])
    ax.set_xlim(-axis_ranges[dim1], axis_ranges[dim1]) # note inverse X axis
    ax.set_ylim(-axis_ranges[dim2], axis_ranges[dim2])

def plot_sub_spreads(times, bayes_spreads, naive_spreads, init_conditions, ax):
    """Plot the bayesian and naive fits to the spread of stars

    Parameters
    ----------
    times : [ntimes] array
        Discrete time steps corresponding to the exact traceback steps
    bayes_spreads : [ntimes] array
        The idealised spherical radius of the position component of the
        bayesian fit
    naive_spreads : [ntimes] array
        The idealised spherical radius of the gaussian fit to the "exact" orbits
    init_conditions : [14] array
        Set of parameters used to construct synthetic data initially
    ax : pyplot axes object
        The axes on which to be plotted
    """
    ax.plot(times, naive_spreads, label="Naive fit")
    ax.plot(times, bayes_spreads, label="Bayes fit")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(
        bottom=0.0, top=max(np.max(naive_spreads), np.max(bayes_spreads))
    )
    if init_conditions is not None:
        init_age = init_conditions[13]
        ax.axvline(
            init_age, ax.get_ylim()[0], ax.get_ylim()[1], color='r', ls='--'
        )
    ax.legend(loc=1)
    ax.set_xlabel("Traceback Time [Myr]")
    ax.set_ylabel("Radius of average spread in XYZ [pc]")

def plot_age_hist(chain, ax, init_conditions=None):
    """Plot a histogram of the ages

    A histogram marginalises over all the parameters yielding a
    bayesian statistical description of the best fitting age

    Parameters
    -----------
    chain : [nsteps, nwalkers, npars] array
        the chain of samples
    ax : pyplot axes object
        the axes on which to plot
    """
    print("In plot_age_hist")
    ax.hist(chain[:,:,-1].flatten(), bins=20)

    ax.set_xlabel("Ages [Myr]")
    ax.set_ylabel("Number of samples")

    if init_conditions is not None:
        init_age = init_conditions[13]
        ax.axvline(
            init_age, ax.get_ylim()[0], ax.get_ylim()[1], color='r', ls='--'
        )

    print("Done most of plot_age_hist")

def plot_sub_age_pdf(times, time_probs, init_conditions, ax):
    """
    Normalise and the plot the age pdf

    Parameters
    ----------
    times : [ntimes] array
        Discrete time steps corresponding to the exact traceback steps
    time_probs : [ntimes] array
        scaled average likelihoods of the samples for each fixed time fit
    init_conditions : [14] array
        Set of parameters used to construct synthetic data initially
    ax : pyplot axes object
        axes on which to be plotted
    """
    normalised_time_probs = time_probs / calc_area(time_probs)
    ax.plot(times, normalised_time_probs)
    ax.set_xlim(times[0],times[-1])
    ax.set_ylim(bottom=0.0, top=1.1*max(normalised_time_probs))
    if init_conditions is not None:
        init_age = init_conditions[13]
        ax.axvline(
            init_age, ax.get_ylim()[0], ax.get_ylim()[1], color='r', ls='--'
        )
    ax.set_xlabel("Traceback Time [Myr]")
    ax.set_ylabel("Age probability")

def plot_quadplots(infile, bayes_spreads=None, naive_spreads=None, time_probs=None,
                   init_conditions=None, dir='', plot_it=False):
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
        bayes_spreads, time_probs = gf.get_bayes_spreads(infile, plot_it=plot_it)
    assert(len(times) == len(naive_spreads))
    assert(len(times) == len(bayes_spreads))

    best_fit_free, chain_free, lnprob_free =\
        gf.fit_group(
            infile, burnin_steps=1000, sampling_steps=1000, plot_it=plot_it
        )

    # Plot spread fits
    plt.clf()
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    f.set_size_inches(20, 20)

    plot_sub_traceback(xyzuvw, xyzuvw_cov, times, 0, 1, ax1)
    plot_sub_spreads(times, bayes_spreads, naive_spreads, init_conditions, ax2)
    #plot_sub_traceback(xyzuvw, xyzuvw_cov, times, 3, 4, ax3)
    plot_age_hist(chain_free, ax3, init_conditions=init_conditions)
    plot_sub_age_pdf(times, time_probs, init_conditions, ax4)
    f.savefig("temp_plot.png")
