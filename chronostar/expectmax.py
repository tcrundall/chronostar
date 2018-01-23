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
import chronostar._overlap as ol
import utils
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

def calc_errors(chain):
    """
    Given a set of aligned (converted?) samples, calculate the median and
    errors of each parameter

    Parameters
    ----------
    chain : [nwalkers, nsteps, npars]
        The chain of samples (in internal encoding)
    """
    npars = chain.shape[-1]     # will now also work on flatchain as input
    flat_chain = np.reshape(chain, (-1, npars))

    conv_chain = np.copy(flat_chain)
    conv_chain[:, 6:10] = 1/conv_chain[:, 6:10]

    return np.array( map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                     zip(*np.percentile(conv_chain, [16,50,84], axis=0))))

def check_convergence(new_best_fit, old_best_fit, new_chain, old_chain, tol=1.0):
    """Check if the last maximisation step yielded consistent fit to new fit

    Convergence is achieved if key values are within 'tol' sigma across the two
    most recent fits.

    Parameters
    ----------
    new_best_fit : [15] array
        paraameters (in external encoding) of the best fit from the new run
    old_best_fit : [15] array
        paraameters (in external encoding) of the best fit from the old run
    new_chain : [nwalkers, nsteps, npars] array
        the sampler chain from the new run
    old_chain : [nwalkers, nsteps, npars] array
        the sampler chain from the old run
    tol : float
        Within how many sigma the values should be

    Returns
    -------
    converged : bool
        If the runs have converged, return true
    """
    return False


def calc_lnoverlaps(group_pars, star_pars, nstars):
    """Find the lnoverlaps given the parameters of a group

    Parameters
    ----------
    group_pars : [npars] array
        Group parameters (internal encoding, 1/dX... no nstars)
    star_pars : dict
        stars: (nstars) high astropy table including columns as
            documented in the Traceback class.
        times : [ntimes] numpy array
            times that have been traced back, in Myr
        xyzuvw : [nstars, ntimes, 6] array
            XYZ in pc and UVW in km/s
        xyzuvw_cov : [nstars, ntimes, 6, 6] array
            covariance of xyzuvw
    nstars : int
        number of stars in traceback

    Returns
    -------
    lnols : [nstars] array
        The log of the overlap of each star with the provided group
    """
    group_mn = group_pars[0:6]
    group_cov = utils.generate_cov(group_pars)
    assert np.min( np.linalg.eigvalsh(group_cov) ) >= 0

    # interpolate star data to modelled age
    age = group_pars[13]
    interp_covs, interp_mns = gf.interp_cov(age, star_pars)
    lnols = ol.get_lnoverlaps(
        group_cov, group_mn, interp_covs, interp_mns, nstars
    )
    return lnols

def calc_membership_probs(star_lnols):
    """Calculate probabilities of membership for a single star from overlaps

    Parameters
    ----------
    star_lnols : [ngroups] array
        The log of the overlap of a star with each group

    Returns
    -------
    star_memb_probs : [ngroups] array
        The probability of membership to each group, normalised to sum to 1
    """
    ngroups = star_lnols.shape[0]
    star_memb_probs = np.zeros(ngroups)

    for i in range(ngroups):
        star_memb_probs[i] = 1. / np.sum(np.exp(star_lnols - star_lnols[i]))

    return star_memb_probs

def expectation(star_pars, groups):
    """Calculate membership probabilities given fits to each group

    Parameters
    ----------
    star_pars : dict
        stars: (nstars) high astropy table including columns as
                    documented in the Traceback class.
        times : [ntimes] numpy array
            times that have been traced back, in Myr
        xyzuvw : [nstars, ntimes, 6] array
            XYZ in pc and UVW in km/s
        xyzuvw_cov : [nstars, ntimes, 6, 6] array
            covariance of xyzuvw

    groups : [ngroups, npars] array
        a fit for each group (in internal form)

    Returns
    -------
    z : [nstars, ngroups] array
        An array designating each star's probability of being a member to each
        group. It is populated by floats in the range (0.0, 1.0) such that
        each row sums to 1.0, each column sums to the expected size of each
        group, and the entire array sums to the number of stars.
    """
    ngroups = groups.shape[0]
    nstars = len(star_pars['xyzuvw'])
    lnols = np.zeros((nstars, ngroups))

    for i, group_pars in enumerate(groups):
        lnols[:,i] = calc_lnoverlaps(group_pars, star_pars, nstars)

    z = np.zeros((nstars, ngroups))
    for i in range(nstars):
        z[i] = calc_membership_probs(lnols[i])

    return z

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
            plot_it=True
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
    star_params = gf.read_stars(infile)
    samples, lnprob = gf.fit_one_group(star_params, nburnin, nsteps)
    return samples, lnprob
