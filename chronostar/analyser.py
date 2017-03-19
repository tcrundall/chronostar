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

def read_sampler(infile='logs/gf_bp_2_3_10_10.pkl'):
    """
    Read in sampler from .pkl file
    """
    chain, lnprob, pos, nfree, nfixed = pickle.load(open(infile))
    nwalkers, nsteps, npars = np.shape(chain)

    return chain, lnprob, pos, nfree, nfixed, nsteps, npars

def realign_samples():
    """
    given a set of samples, realign sample by finding the permutation
    of each sample's group parameters such that the groups in the 
    permuted sample are the closest they can be to the
    groups represented by the sample with the highest lnprob
    """
    return 0

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
