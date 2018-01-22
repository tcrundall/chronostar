"""
a module for implementing the expectation-maximisation algorithm
in order to fit a multi-gaussian mixture model of moving groups' origins
to a data set of stars tracedback through XYZUVW

todo:
    - implement average error cacluation in lnprobfunc
"""
from __future__ import print_function, division

import numpy as np
import groupfitter as gf
import matplotlib.pyplot as plt
import pdb          # for debugging
import corner       # for pretty corner plots
import pickle       # for dumping and reading data
# for permuting samples when realigning
from sympy.utilities.iterables import multiset_permutations 
try:
    import astropy.io.fits as pyfits
except:
    import pyfits

try:
    import _overlap as overlap #&TC
except:
    print("overlap not imported, SWIG not possible. Need to make in directory...")
from emcee.utils import MPIPool

try:                # don't know why we use xrange to initialise walkers
    xrange
except NameError:
    xrange = range

def ix_fst(array, ix):
    if array is None:
        return None
    else:
        return array[ix]

def ix_snd(array, ix):
    if array is None:
        return None
    else:
        return array[:,ix]

def maximise(infile, ngroups, z=None, init_conditions=None, burnin_steps=500,
             sampling_steps=1000):
    """Given membership probabilities, maximise the parameters of each group model

    Parameters
    ----------
    infile : str
        Name of the traceback file being fitted to

    ngroups : int
        Number of groups to be fitted to the traceback orbits

    z : [nstars, ngroups] array
        An array designating each star's probability of being a member to each
        group. It is populated by floats in the range (0.0, 1.0) such that
        each row sums to 1.0, each column sums to the expected size of each
        group, and the entire array sums to the number of stars.

    init_conditions : [ngroups, npars] array
        The initial conditions for the groups, encoded in the 'internal' manner
        (that is, 1/dX, 1/dY etc. and no star count)

    burnin_steps : int
        number of steps during burnin phase

    sampling_steps : int
        number of steps during sampling phase

    Returns
    -------
    best_fits : [ngroups, npars] array
        The best fitting parameters for each group
    chains : [ngroups, nwalkers, nsteps, npars] array
        The final chain for each group's fit
    lnprobs : [ngroups, nwalkers, nsteps] array
        The sampler.probability array for each group's fit

    TODO: Have some means to enforce fixed ages
    """
    NPARS = 14
    best_fits = np.zeros((ngroups, NPARS))
    chains = None
    lnprobs = None

    for i in range(ngroups):
        best_fit, chain, lnprob = gf.fit_group(
            infile, z=ix_snd(z, i), init_pars=ix_fst(init_conditions, i),
            burnin_steps=burnin_steps, sampling_steps=sampling_steps,
        )
        best_fits[i] = best_fit
        if chains is None:
            dims = np.append(ngroups, chain.shape)
            chains = np.zeros(dims)
        if lnprobs is None:
            dims = np.append(ngroups, lnprob.shape)
            lnprobs = np.zeros(dims)

        chains[i] = chain
        lnprobs[i] = lnprob
    return best_fits, chains, lnprobs

def run_fit(infile, nburnin, nsteps, ngroups=1):
    """
    Entry point for module. Given the traceback of stars, fit a set number
    of groups to the data. Left unfulfilled atm since only fitting a single
    group.
    """
    star_params = read_stars(infile)
    samples, lnprob = fit_one_group(star_params, nburnin, nsteps)
    return samples, lnprob
