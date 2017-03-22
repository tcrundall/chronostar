"""
This program analyses the output of an emcee run (Monte-Carlo Markov Chain).
Given any output of group_fitter, this module will load the samples, derive
and include parametersas necessary, convert parameters to physical values,
produce lnprob plots of the walkerss as well as corner plots of arbitrary
combinations of parameters.

It can also extract the median values of each parameters with upper and lower
error ranges.

This module can also generate output for a future run of group_fitter. Either 
by retrieving the tidied final state of a run, or by generating initial
conditions from the median values and erros of parameters.

It is conceivable that a full algorithmic fit would alternate
between runs of group_fitter generating samples, and runs of analyser, tidying
and neatening, and consolidating
"""

from __future__ import print_function, division

import pdb          # for debugging
import corner       # for pretty corner plots
import pickle       # for dumping and reading data
import numpy as np
from sympy.utilities.iterables import multiset_permutations

def read_sampler(infile='logs/gf_bp_2_3_10_10.pkl'):
    """
    Read in sampler from .pkl file
    """
    chain, lnprob, pos, nfree, nfixed = pickle.load(open(infile))
    nwalkers, nsteps, npars = np.shape(chain)

    return chain, lnprob, pos, nfree, nfixed, nsteps, npars

def realign_samples(flatchain, flatlnprob, nfree, nfixed, npars):
    """
    given a set of samples, realign sample by finding the permutation
    of each sample's group parameters such that the groups in the 
    permuted sample are the closest they can be to the
    groups represented by the sample with the highest lnprob
    """
    # Initialising array for realigned samples, extra space for weight
    # to be derived
    # GET MIKE TO HELP WORK OUT HOW TO SIMPLY APPEND EXTRA PARAMETER
    # USING NP.APPEND OR SOME SHIT
    realigned_samples = np.zeros((np.shape(flatchain)[0],
                                  np.shape(flatchain)[1]+1))
    ngroups = nfree+nfixed

    # no need to realign anything if there's only the one group
    if ngroups == 0:
        return flatchain

    for i, sample in enumerate(flatchain):
        weights = sample[-(ngroups-1):]
        derived_weight = 1 - np.sum(weights)
        realigned_samples[i] = np.append(sample, derived_weight)

    best_ix = np.argmax(flatlnprob)
    best_sample = realigned_samples[best_ix]

    permute_count = 0

    for i, sample in enumerate(realigned_samples):
        realigned_sample = permute(sample, best_sample, nfree, nfixed)
        if np.array_equal(realigned_samples[i],realigned_sample):
            permute_count += 1
        realigned_samples[i] = realigned_sample
    
    return realigned_samples, permute_count

def permute(sample, best_sample, nfree, nfixed):
    """
    Takes a sample and the best_sample (the one with the highest
    lnprob) and returns the perumtation of sample which is the closest
    to best_sample.

    Input samples should already have derived weight appeneded.

    The reason this is necessary is because, if sample has more than one
    free group, there is nothing distinguishing which group is character-
    ised by which parameters
    """
    npars_wo_amp = 14
    assert(np.size(best_sample) == npars_wo_amp * nfree + (nfree + nfixed))

    # Reshape the parameters for free groups into an array with each row
    # corresponding to a specific group
    free_groups = np.reshape(  sample[:npars_wo_amp * nfree],(nfree,-1))
    best_fgs = np.reshape(best_sample[:npars_wo_amp * nfree],(nfree,-1))

    if (nfixed != 0):
        free_amps   = sample[-(nfree + nfixed):-nfixed]
        fixed_amps  = sample[-nfixed:]
        best_fas = best_sample[-(nfree + nfixed):-nfixed]
        best_xas = best_sample[-nfixed:]
    else:
        # Special behaviour if all groups are free
        free_amps   = sample[-nfree:]
        fixed_amps  = sample[:0]        # an empty array
        best_fas = best_sample[-nfree:]
        best_xas = best_sample[:0]      # an empty array

    # try using the np.fromfunction here?
    # This is a distance matrix where Dmat[i,j] is the metric distance
    # between the ith free_group and the jth best_free_group
    Dmat = np.zeros((nfree,nfree))
    for i in range(nfree):
        for j in range(nfree):
            Dmat[i,j] = group_metric(free_groups[i], best_fgs[j])

    # Generate a list of all possible permutations of the matrix rows
    ps = [p for p in multiset_permutations(range(nfree))]

    # Find the matrix permutation which yields the smallest trace
    # This permutation will correspond to the rearrangement of free_groups
    # which is most similar to the best_free_groups
    traces = [np.trace(Dmat[p]) for p in ps]
    best_perm = ps[np.argmin(traces)]

    # Recombine the rearranged free_groups,
    # note that the amplitudes to the free_gorups must also be
    # rearranged with the identical permutation
    perm_sample = np.append(np.append(free_groups[best_perm],
                                      free_amps[best_perm]),
                            fixed_amps)
    try:
        assert(np.size(perm_sample) == np.size(sample)),\
                "Wrong size...\n{}\n{}".format(sample, perm_sample)
    except:
        pdb.set_trace()

    return perm_sample

def group_metric(group1, group2):
    """
    Calculates the metric distance between two groups.
    Inputs are two np.arrays of size 14.
    Note that the group inputs are raw parametes, that is stds are
    parametrised as 1/std in the emcee chain so must be inverted
    """
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

def calc_bet_fit():
    """
    Given a set of aligned samples, calculate the median and errors of each
    parameter
    """
    return 0

def plot_lnprob():
    """
    Generate lnprob and lnprob.T plots, save them to file as necessary
    """
    return 0

def plot_corner():
    """
    Generate corner plots with dynamically generated parameter list
    e.g. ONly plotting stds or ages, or weights, or any combinations
    """
    return 0
