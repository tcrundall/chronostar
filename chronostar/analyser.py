"""
"""

from __future__ import print_function, division

import corner       # for pretty corner plots
import pickle       # for dumping and reading data
import numpy as np
from sympy.utilities.iterables import multiset_permutations


def group_metric(group1, group2):
    """
    Calculates the metric distance between two groups.
    Inputs are two np.arrays of size 14.
    Note that the group inputs are raw parametes, that is stds are
    parametrised as 1/std in the emcee chain so must be inverted

    TO DO: USE BOOLEAN MASKS TO EXTRACT PARAMETERS IN A NEATER MANNER
    """

    # REWRITE USING BOOLEAN MASKS
    means1 = group1[:6];      means2 = group2[:6]
    stds1  = 1/group1[6:10];  stds2  = 1/group2[6:10]
    corrs1 = group1[10:13];   corrs2 = group2[10:13]
    age1   = group1[13];      age2   = group2[13]

    total_dist = 0
    for i in range(3):
        total_dist += (means1[i] - means2[i])**2 /\
                        (stds1[i]**2 + stds2[i]**2)

    for i in range(3,6):
        total_dist += (means1[i] - means2[i])**2 /\
                        (stds1[3]**2 + stds2[3]**2)

    for i in range(4):
        total_dist += (np.log(stds1[i] / stds2[i]))**2

    for i in range(3):
        total_dist += (corrs1[i] - corrs2[i])**2

    total_dist += (np.log(age1/age2))**2

    return np.sqrt(total_dist)

def calc_best_fit(flat_samples):
    """
    Given a set of aligned (converted?) samples, calculate the median and
    errors of each parameter
    """
    return np.array( map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                     zip(*np.percentile(flat_samples, [16,50,84], axis=0))) )

def plot_corner(nfree, nfixed, converted_samples, lnprob,
                means=False, stds=False,    corrs=False,
                ages=False,  weights=False, tstamp='nostamp'):
    """LEGACY

    Generate corner plots with dynamically generated parameter list
    e.g. ONly plotting stds or ages, or weights, or any combinations
    """
    # Checking at least one value is being plotted
    if not (means or stds or corrs or ages or weights):
        print("Need to include a boolean flag for desired parameters")
        return 0

    labels = generate_labels(nfree, nfixed)
    param_mask = generate_param_mask(nfree, nfixed, means, stds,
                                     corrs, ages, weights)

    best_ix = np.argmax(lnprob.flatten())
    best_sample = converted_samples[best_ix] 
    # fig = corner.corner(converted_samples[:, np.where(param_mask)],
    #                     truths = best_sample[np.where(param_mask)],
    #                     labels =      labels[np.where(param_mask)] )
    fig = corner.corner(converted_samples[:, np.where(param_mask)][:,0],
                        truths = best_sample[np.where(param_mask)],
                        labels =      labels[np.where(param_mask)] )

    file_name = "plots/{}_corner_{}_{}_{}.png".format(tstamp, nfree, nfixed,
                                                      lnprob.shape[1])

    pdb.set_trace()

    fig.savefig(file_name)
    fig.clf()
    return 0

def write_results(nsteps, ngroups, bg_groups, bf, tstamp, nsaved=0, bw=None,
                  infile=None, info=""):
    """LEGACY

    Saves the results of a fit to file.
    """
    # Generate a label for all of our groups 
    labels = generate_labels(ngroups, 0)

    with open( "logs/{}_{}_{}_{}.txt".\
              format(tstamp, ngroups, bg_groups, nsteps), 'w') as f:
        f.write("Log of output from bp with {} groups, {} bg_groups and {} "
                "sampling steps,\n".format(ngroups, bg_groups, nsteps) )
        if infile:
            f.write("Input data: {}\n".format(infile))
        f.write("\n")
        f.write(info)
        f.write("\n")
        f.write("______ MOVING GROUP ______\n")
        for i in range(len(labels)):
            f.write("{:8}: {:> 7.2f}  +{:>5.2f}  -{:>5.2f}\n"\
                    .format(labels[i], bf[i][0], bf[i][1], bf[i][2]))
        for i in range(nsaved):
            f.write("{:8}: {:> 7.2f}  +{:>5.2f}  -{:>5.2f}\n"\
                    .format("weight", bf[len(labels)+i][0],
                            bf[len(labels)+i][1], bf[len(labels)+i][2]))
        if bw is not None:
            f.write("{:8}: {:> 7.2f}  +{:>5.2f}  -{:>5.2f}\n"\
                    .format("width", bw[0][0], bw[0][1], bw[0][2]))

def calc_naive_spread(xs, ys):
    """Calculate the XY spread of stars at a single snapshot

    Returns the geometric mean of the eigenvalues for the associated 2x2
    covariance matrix
    """
    approx_cov = np.cov(np.vstack((xs, ys)))
    ellipse_axes = np.sqrt(np.linalg.eigvalsh(approx_cov))
    return np.prod(ellipse_axes)**0.5

def get_naive_spreads(xyzuvw):
    """Calculate the occupied volume of a group with a naive method

    Calculates the effective width (standard deviation) of the XY error
    ellipse associated with the covariance matrix that fits the xy data.

    Parameters
    ----------
    xyzuvw : [nstars, ntimes, 6] np array

    Output
    ------
    naive_spreads : [ntimes] np array
        the measure of the occupied volume of a group at each time
    """
    ntimes = xyzuvw.shape[1]
    naive_spreads = np.zeros(ntimes)

    for i in range(ntimes):
        xs = xyzuvw[:, i, 0]
        ys = xyzuvw[:, i, 1]
        naive_spreads[i] = calc_naive_spread(xs, ys)

    return naive_spreads


