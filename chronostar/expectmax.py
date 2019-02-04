"""
a module for implementing the expectation-maximisation algorithm
in order to fit a multi-gaussian mixture model of moving groups' origins
to a data set of stars tracedback through XYZUVW

todo:
    - implement average error cacluation in lnprobfunc
"""
from __future__ import print_function, division

import pdb
import sys
from distutils.dir_util import mkpath
import logging
import numpy as np
import os
import pickle
import random

import chronostar.synthesiser as syn
import chronostar.traceorbit as torb

try:
    import matplotlib as mpl
    # prevents displaying plots from generation from tasks in background
    mpl.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not imported")
    pass

import chronostar.transform as tf
import groupfitter as gf

def ix_fst(array, ix):
    """Helper function to index array by first axis that could be None"""
    if array is None:
        return None
    else:
        return array[ix]


def ix_snd(array, ix):
    """Helper function to index array by second axis that could be None"""
    if array is None:
        return None
    else:
        return array[:, ix]


def calcMedAndSpan(chain, perc=34, sphere=True):
    """
    Given a set of aligned samples, calculate the 50th, (50-perc)th and
     (50+perc)th percentiles.

    Parameters
    ----------
    chain : [nwalkers, nsteps, npars]
        The chain of samples (in internal encoding)
    perc: integer {34}
        The percentage from the midpoint you wish to set as the error.
        The default is to take the 16th and 84th percentile.
    sphere: Boolean {True}
        Currently hardcoded to take the exponent of the logged
        standard deviations. If sphere is true the log stds are at
        indices 6, 7. If sphere is false, the log stds are at
        indices 6:10.

    Returns
    -------
    _ : [npars,3] float array
        For each paramter, there is the 50th, (50+perc)th and (50-perc)th
        percentiles
    """
    npars = chain.shape[-1]  # will now also work on flatchain as input
    flat_chain = np.reshape(chain, (-1, npars))

    # conv_chain = np.copy(flat_chain)
    # if sphere:
    #     conv_chain[:, 6:8] = np.exp(conv_chain[:, 6:8])
    # else:
    #     conv_chain[:, 6:10] = np.exp(conv_chain[:, 6:10])

    # return np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
    #                     zip(*np.percentile(conv_chain,
    #                                        [50-perc,50,50+perc],
    #                                        axis=0))))
    return np.array(map(lambda v: (v[1], v[2], v[0]),
                        zip(*np.percentile(flat_chain,
                                           [50-perc, 50, 50+perc],
                                           axis=0))))


def checkConvergence(old_best_fits, new_chains,
                     perc=35):
    """Check if the last maximisation step yielded is consistent to new fit

    Note, percentage raised to 35 (from 25) as facing issues establishing
    convergence despite 100+ iterations for components with ~10 stars.
    Now, the previous best fits must be within the 70% range (i.e. not
    fall within the bottom 15th or top 15th percentiles in any parameter).

    TODO: incorporate Z into this convergence checking. e.g.
    np.allclose(z_prev, z, rtol=1e-2)

    Convergence is achieved if previous key values fall within +/-"perc" of
    the new fits.

    Parameters
    ----------
    old_best_fits : [ncomp] list of synthesiser Group objects
    new_chain : list of ([nwalkers, nsteps, npars] array) with ncomp elements
        the sampler chain from the new run
    perc : int (0, 50)
        the percentage distance that previous values must be within current
        values.

    Returns
    -------
    converged : bool
        If the runs have converged, return true
    """
    each_converged = []

    for old_best_fit, new_chain in zip(old_best_fits, new_chains):
        errors = calcMedAndSpan(new_chain, perc=perc)
        upper_contained =\
            old_best_fit.getInternalSphericalPars() < errors[:, 1]
        lower_contained =\
            old_best_fit.getInternalSphericalPars() > errors[:, 2]

        each_converged.append(
            np.all(upper_contained) and np.all(lower_contained))

    return np.all(each_converged)


def calcMembershipProbs(star_lnols):
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


def background6DLogOverlap(star_mean, bg_6dhist):
    """
    Approximates density of Gaia catalogue at `star_mean`
    :param star_mean:
    :param bg_6dhist:
    :return:
    """
    pass


def backgroundLogOverlap(star_mean, bg_hists, correction_factor=1.):
    """Calculate the 'overlap' of a star with the background desnity of Gaia

    We assume the Gaia density is approximately constant over the size scales
    of a star's uncertainty, and so approximate the star as a delta function
    at it's central estimate(/mean)

    Parameters
    ----------
    star_mean : [6] float array
        the XYZUVW central estimate of a star, XYZ in pc and UVW in km/s

    bg_hists : 6*[[nbins],[nbins+1]] list
        A collection of histograms desciribing the phase-space density of
        the Gaia catalogue in the vicinity of associaiton in quesiton.
        For each of the 6 dimensions there is an array of bin values and
        an array of bin edges.

        e.g. bg_hists[0][1] is an array of floats describing the bin edges
        of the X dimension 1D histogram, and bg_hists[0][0] is an array of
        integers describing the star counts in each bin

    correction_factor : positive float {1.}
        artificially amplify the Gaia background density to account for
        magnitude correctness
    """
    # get the total area under a histogram
    n_gaia_stars = np.sum(bg_hists[0][0])

    ndim = 6
    lnol = 0
    for dim_ix in range(ndim):
        # evaluate the density: bin_height / bin_width / n_gaia_stars
        bin_width = bg_hists[dim_ix][1][1] - bg_hists[dim_ix][1][0]
        lnol += np.log(
            bg_hists[dim_ix][0][np.digitize(star_mean[dim_ix],
                                            bg_hists[dim_ix][1]) - 1]\
        )
        lnol -= bin_width
        lnol -= np.log(n_gaia_stars)

    # Renormalise such that the combined 6D histogram has a hyper-volume
    # of n_gaia_stars
    # lnol -= 5 * np.log(n_gaia_stars)
    # lnol += np.log(n_gaia_stars)
    lnol += np.log(n_gaia_stars*correction_factor)
    return lnol


def backgroundLogOverlaps(xyzuvw, bg_hists, correction_factor=1.0):
    """Calculate the 'overlaps' of stars with the background desnity of Gaia

    We assume the Gaia density is approximately constant over the size scales
    of a star's uncertainty, and so approximate the star as a delta function
    at it's central estimate(/mean)

    Parameters
    ----------
    xyzuvw: [nstars, 6] float array
        the XYZUVW central estimate of a star, XYZ in pc and UVW in km/s

    bg_hists : 6*[[nbins],[nbins+1]] list
        A collection of histograms desciribing the phase-space density of
        the Gaia catalogue in the vicinity of associaiton in quesiton.
        For each of the 6 dimensions there is an array of bin values and
        an array of bin edges.

        e.g. bg_hists[0][1] is an array of floats describing the bin edges
        of the X dimension 1D histogram, and bg_hists[0][0] is an array of
        integers describing the star counts in each bin

    correction_factor : positive float {1.}
        artificially amplify the Gaia background density to account for
        magnitude correctness

    Returns
    -------
    bg_ln_ols: [nstars] float array
        the overlap with each star and the flat-ish background field
        distribution
    """
    bg_ln_ols = np.zeros(xyzuvw.shape[0])
    for i in range(bg_ln_ols.shape[0]):
        bg_ln_ols[i] = backgroundLogOverlap(
            xyzuvw[i], bg_hists, correction_factor=correction_factor
        )
    return bg_ln_ols


def getAllLnOverlaps(star_pars, groups, old_z=None, bg_ln_ols=None,
                     inc_posterior=False, amp_prior=None):
    """
    Get the log overlap integrals of each star with each component

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

    groups : [ngroups] syn.Group object list
        a fit for each group (in internal form)

    old_z : [nstars, ngroups (+1)] float array {None}
        Only used to get weights (amplitudes) for each fitted component.
        Tracks membership probabilities of each star to each group. Each
        element is between 0.0 and 1.0 such that each row sums to 1.0
        exactly.
        If bg_hists are also being used, there is an extra column for the
        background. Note that it is not used in this function

    bg_ln_ols : [nstars] float array
        The overlap the stars have with the (fixed) background distribution

    inc_posterior: bool {False}
        If true, includes prior on groups into their relative weightings

    amp_prior: int {None}
        If set, forces the combined ampltude of Gaussian components to be
        at least equal to `amp_prior`

    Returns
    -------
    lnols: [nstars, ngroups (+1)] float array
        The log overlaps of each star with each component, optionally
        with the log background overlaps appended as the final column
    """
    nstars = len(star_pars['xyzuvw'])
    ngroups = len(groups)
    using_bg = bg_ln_ols is not None

    lnols = np.zeros((nstars, ngroups + using_bg))

    if old_z is None:
        old_z = np.ones((nstars, ngroups)) / ngroups


    weights = old_z[:,:ngroups].sum(axis=0)

    # Optionally scale each weight by the component prior, then rebalance
    # so total expected stars across all components is unchanged
    if inc_posterior:
        group_lnpriors = np.zeros(ngroups)# &TC
        for i, group in enumerate(groups):
            group_lnpriors[i] = gf.lnAlphaPrior(
                group.getInternalSphericalPars(),
                z=old_z
            )
        ngroup_stars = weights.sum()
        weights *= np.exp(group_lnpriors)
        weights = weights / weights.sum() * ngroup_stars

    # Optionally scale each weight such that the total expected stars
    # is equal to or greater than `amp_prior`
    if amp_prior:
        if weights.sum() < amp_prior:
            weights *= amp_prior / weights.sum()

#    # except:
#    logging.info("_____ DEBUGGGING _____")
#    logging.info("ngroups: {}".format(ngroups))
#    logging.info("old_z shape: {}".format(old_z.shape))
#    logging.info("weights shape: {}".format(weights.shape))
#    logging.info("weights: {}".format(weights))
#    import pdb; pdb.set_trace()

    for i, group in enumerate(groups):
        # weight is the amplitude of a component, proportional to its expected
        # total of stellar members
        #weight = old_z[:,i].sum()
        # threshold = nstars/(2. * (ngroups+1))
        # if weight < threshold:
        #     logging.info("!!! GROUP {} HAS LESS THAN {} STARS, weight: {}".\
        #         format(i, threshold, weight)
        # )
        group_pars = group.getInternalSphericalPars()
        lnols[:, i] =\
            np.log(weights[i]) +\
                gf.getLogOverlaps(group_pars, star_pars)
            # gf.lnlike(group_pars, star_pars,
            #                            old_z, return_lnols=True) #??!??!?!

    # insert one time calculated background overlaps
    if using_bg:
        lnols[:,-1] = bg_ln_ols
    return lnols


def calcBIC(star_pars, ncomps, lnlike):
    """Calculates the Bayesian Information Criterion

    A simple metric to judge whether added components are worthwhile
    """
    nstars = len(star_pars['xyzuvw'])
    n = nstars * 7 # 6 for phase space origin and 1 for age
    k = ncomps * 8 # 6 for central estimate, 2 for dx and dv, 1 for age
    return np.log(n)*k - 2 * lnlike


def expectation(star_pars, groups, old_z=None, bg_ln_ols=None,
                inc_posterior=False, amp_prior=None):
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

    groups : [ngroups] syn.Group object list
        a fit for each group (in internal form)

    old_z : [nstars, ngroups (+1)] float array
        Only used to get weights (amplitudes) for each fitted component.
        Tracks membership probabilities of each star to each group. Each
        element is between 0.0 and 1.0 such that each row sums to 1.0
        exactly.
        If bg_hists are also being used, there is an extra column for the
        background. However it is not used in this context

    bg_ln_ols : [nstars] float array
        The overlap the stars have with the (fixed) background distribution

    Returns
    -------
    z : [nstars, ngroups] array
        An array designating each star's probability of being a member to
        each group. It is populated by floats in the range (0.0, 1.0) such
        that each row sums to 1.0, each column sums to the expected size of
        each group, and the entire array sums to the number of stars.
    """
    ngroups = len(groups)
    nstars = len(star_pars['xyzuvw'])

    using_bg = bg_ln_ols is not None

    # if no z provided, assume perfectly equal membership
    if old_z is None:
        old_z = np.ones((nstars, ngroups + using_bg))/(ngroups + using_bg)

    lnols = getAllLnOverlaps(star_pars, groups, old_z, bg_ln_ols,
                             inc_posterior=inc_posterior, amp_prior=amp_prior)

    z = np.zeros((nstars, ngroups + using_bg))
    for i in range(nstars):
        z[i] = calcMembershipProbs(lnols[i])
    if np.isnan(z).any():
        logging.info("!!!!!! AT LEAST ONE MEMBERSHIP IS 'NAN' !!!!!!")
        z[np.where(np.isnan(z))] = 0.
        # import pdb; pdb.set_trace()
    return z


def getPointsOnCircle(npoints, v_dist=20, offset=False):
    """
    Little tool to found coordinates of equidistant points around a circle

    Used to initialise UV for the groups.
    :param npoints:
    :return:
    """
    us = np.zeros(npoints)
    vs = np.zeros(npoints)
    if offset:
        init_angle = np.pi / npoints
    else:
        init_angle = 0.

    for i in range(npoints):
        us[i] = v_dist * np.cos(init_angle + 2 * np.pi * i / npoints)
        vs[i] = v_dist * np.sin(init_angle + 2 * np.pi * i / npoints)

    return np.vstack((us, vs)).T


def getInitialGroups(ngroups, xyzuvw, offset=False, v_dist=10.):
    """
    Generate the parameter list with which walkers will be initialised

    Parameters
    ----------
    ngroups: int
        number of groups
    xyzuvw: [nstars, 6] array
        the mean measurement of stars
    offset : (boolean {False})
        If set, the gorups are initialised in the complementary angular
        positions
    v_dist: float
        Radius of circle in UV plane along which groups are initialsed

    Returns
    -------
    groups: [ngroups] synthesiser.Group object list
        the parameters with which to initialise each group's emcee run
    """
    groups = []

    # if only fitting one group, simply initialise walkers about the
    # mean of the data set
    if ngroups == 1:
        v_dist = 0

    mean = np.mean(xyzuvw, axis=0)[:6]
    logging.info("Mean is\n{}".format(mean))
#    meanXYZ = np.array([0.,0.,0.])
#    meanW = 0.
    dx = 100.
    dv = 15.
    age = 3.
    # group_pars_base = list([0, 0, 0, None, None, 0, np.log(50),
    #                         np.log(5), 3])
    pts = getPointsOnCircle(npoints=ngroups, v_dist=v_dist, offset=offset)
    logging.info("Points around circle are:\n{}".format(pts))

    for i in range(ngroups):
        mean_w_offset = np.copy(mean)
        mean_w_offset[3:5] += pts[i]
        logging.info("Group {} has init UV of ({},{})".\
                    format(i, mean_w_offset[3], mean_w_offset[4]))
        group_pars = np.hstack((mean_w_offset, dx, dv, age))
        group = syn.Group(group_pars, sphere=True, starcount=False)
        groups.append(group)

    return groups

def decomposeGroup(group, young_age=None, old_age=None, age_offset=4):
    """
    Takes a group object and splits it into two components offset by age.

    Parameters
    ----------
    group: synthesiser.Group instance
        the group which is to be decomposed

    Returns
    -------
    all_init_pars: [2, npars] array
        the intenralised parameters with which the walkers will be
        initiallised
    sub_groups: [2] list of Group instances
        the group objects of the resulting decomposition
    """
    internal_pars = group.getInternalSphericalPars()
    mean_now = torb.traceOrbitXYZUVW(group.mean, group.age, single_age=True)
    ngroups = 2

    sub_groups = []

    if (young_age is None) and (old_age is None):
        young_age = max(1e-5, group.age - age_offset)
        old_age = group.age + age_offset

    ages = [young_age, old_age]
    for age in ages:
        mean_then = torb.traceOrbitXYZUVW(mean_now, -age, single_age=True)
        group_pars_int = np.hstack((mean_then, internal_pars[6:8], age))
        sub_groups.append(syn.Group(group_pars_int, sphere=True,
                                    internal=True, starcount=False))
    all_init_pars = [sg.getInternalSphericalPars() for sg in sub_groups]

    return all_init_pars, sub_groups


def getOverallLnLikelihood(star_pars, groups, bg_ln_ols, return_z=False,
                           inc_posterior=False):
    """
    Get overall likelihood for a proposed model.

    Evaluates each star's overlap with every component and background
    If only fitting one group, inc_posterior does nothing

    Parameters
    ----------
    star_pars : (dict)
    groups : [ngroups] list of Synthesiser.Group objects
    z : [nstars, ngroups] float array
        membership array
    bg_ln_ols : [nstars] float array
        the overlap each star has with the provided background density
        distribution

    Returns
    -------
    overall_lnlikelihood : float
    """
    z = expectation(star_pars, groups, None, bg_ln_ols,
                    inc_posterior=inc_posterior)
    all_ln_ols = getAllLnOverlaps(star_pars, groups, z, bg_ln_ols,
                                  inc_posterior=inc_posterior)

    # multiplies each log overlap by the star's membership probability
    # import pdb; pdb.set_trace()
    weighted_lnols = np.einsum('ij,ij->ij', all_ln_ols, z)
    if return_z:
        return np.sum(weighted_lnols), z
    else:
        return np.sum(weighted_lnols)


def maximisation(star_pars, ngroups, z, burnin_steps, idir,
                 all_init_pars, all_init_pos=None, plot_it=False, pool=None,
                 convergence_tol=0.25, ignore_dead_comps=False):
    """
    Performs the 'maximisation' step of the EM algorithm

    all_init_pars must be given in 'internal' form, that is the standard
    deviations must be provided in log form.

    Parameters
    ----------
    ignore_dead_comps : bool {False}
        if componennts have fewer than 2(?) expected members, skip them

    :param star_pars:
    :param ngroups:
    :param z:
    :param burnin_steps:
    :param idir:
    :param all_init_pars:
    :param all_init_pos: .... not sure how to handle this
    :param plot_it:
    :param pool:
    :param convergence_tol:
    :return:
    """
    DEATH_THRESHOLD = 2.1

    new_groups = []
    all_samples = []
    all_lnprob = []
    # success_mask = np.array(ngroups*[True])
    success_mask = list(np.arange(ngroups))
    if all_init_pos is None:
        all_init_pos = ngroups * [None]

    for i in range(ngroups):
        logging.info("........................................")
        logging.info("          Fitting group {}".format(i))
        logging.info("........................................")

        # --- MARUSA TO EDIT -----
        # IF GROUP IS FAILURE, SKIP
        # nstarspergroup = np.sum(z[:,i])

        # Might need Tim to do this... since unclear how to store 'placeholder blanks' values

        gdir = idir + "group{}/".format(i)
        mkpath(gdir)
        # pdb.set_trace()
        # If component has too few stars, skip fit, and use previous best walker
        if ignore_dead_comps and (np.sum(z[:,i]) < DEATH_THRESHOLD):
            logging.info("Skipped component {} with nstars {}".format(i, np.sum(z[:,i])))
            success_mask.remove(i)
        else:
            best_fit, chain, lnprob = gf.fitGroup(
                xyzuvw_dict=star_pars, burnin_steps=burnin_steps,
                plot_it=plot_it, pool=pool, convergence_tol=convergence_tol,
                plot_dir=gdir, save_dir=gdir, z=z[:, i],
                init_pos=all_init_pos[i],
                init_pars=all_init_pars[i],
            )
            logging.info("Finished fit")
            logging.info("Best group (internal) pars:\n{}".format(best_fit))

            final_pos = chain[:, -1, :]

            logging.info("With age of: {:.3} +- {:.3} Myr".\
                        format(np.median(chain[:,:,-1]), np.std(chain[:,:,-1])))
            new_group = syn.Group(best_fit, sphere=True, internal=True,
                                  starcount=False)
            new_groups.append(new_group)
            np.save(gdir + "best_group_fit.npy", new_group)
            np.save(gdir + 'final_chain.npy', chain)
            np.save(gdir + 'final_lnprob.npy', lnprob)
            all_samples.append(chain)
            all_lnprob.append(lnprob)

            success_mask.append(i)

            # record the final position of the walkers for each group
            all_init_pos[i] = final_pos

    np.save(idir + 'best_groups.npy', new_groups)

    return new_groups, all_samples, all_lnprob, all_init_pos,\
           np.array(success_mask)


def checkStability(star_pars, best_groups, z, bg_ln_ols=None):
    """
    Checks if run has encountered problems

    Common problems include: a component losing all its members, lnprob
    return nans, a membership listed as nan

    Paramters
    ---------
    star_pars : dict
        Contains XYZUVW kinematics of stars
    best_groups : [ngroups] list of Synthesiser.Group objects
        The best fits (np.argmax(chain)) for each component from the most
        recent run
    z : [nstars, ngroups] float array
        The membership array from the most recent run
    bg_ln_ols : [nstars] float array
        The overlap the stars have with the (fixed) background distribution
    """
    ngroups = len(best_groups)
    stable = True
    if np.min(np.sum(z[:,:ngroups], axis=0)) <= 2.:
        logging.info("ERROR: A component has less than 2 members")
        stable = False
    if not np.isfinite(getOverallLnLikelihood(star_pars,
                                              best_groups,
                                              bg_ln_ols)):
        logging.info("ERROR: Posterior is not finite")
        stable = False
    if not np.isfinite(z).all():
        logging.info("ERROR: At least one membership is not finite")
        stable = False
    return stable


def fitManyGroups(star_pars, ngroups, rdir='', init_z=None,
                  origins=None, pool=None, init_with_origin=False,
                  init_groups=None, init_weights=None,
                  offset=False,  bg_hist_file='', correction_factor=1.0,
                  inc_posterior=False, burnin=1000, bg_dens=None,
                  bg_ln_ols=None, ignore_dead_comps=False):
    """
    Entry point: Fit multiple Gaussians to data set

    Parameters
    ----------
    star_pars: dict
        'xyzuvw': [nstars, 6] numpy array
            the xyzuvw mean values of each star, calculated from astrometry
        'xyzuvw_cov': [nstars, 6, 6] numpy array
            the xyzuvw covarince values of each star, calculated from
            astrometry
        'table': Astropy table (sometimes None)
            The astrometry from which xyzuvw values are calculated. Can
            optionally include more information, like star names etc.
    ngroups: int
        the number of groups to be fitted to the data
    rdir: String {''}
        The directory in which all the data will be stored and accessed
        from
    init_z: [nstars, ngroups] array {None} [UNIMPLEMENTED]
        If some members are already known, the initialsiation process
        could use this.
    origins: [ngroups] synthetic Group
    pool: MPIPool object {None}
        the pool of threads to be passed into emcee
    use_background: Bool {False}
        If set, will use histograms based on Gaia data set to compare
        association memberships to the field. Assumes file is in [rdir]
    correction_factor: float {15.3}
        unique to BPMG, calculated by extrapolating the Gaia catalogue
        star count per magnitude if Gaia were sensitive enough to pick up
        the faintest BPMG star
    bg_hist_file: string
        direct path to histogram file being used to inform background
    ignore_dead_comps: bool {False}
        order groupfitter to skip maximising if component has less than...
        2(?) expected members


    Return
    ------
    final_groups: [ngroups] list of synthesiser.Group objects
        the best fit for each component
    final_med_errs: [ngroups, npars, 3] array
        the median, -34 perc, +34 perc values of each parameter from
        each final sampling chain
    z: [nstars, ngroups] array
        membership probabilities

    TODO: Generalise interventions for more than 2 groups
    TODO: Allow option with which step to start with
    """
    # setting up some constants
    BURNIN_STEPS = burnin
    SAMPLING_STEPS = 5000
    C_TOL = 0.5
    nstars = star_pars['xyzuvw'].shape[0]

    logging.info("Fitting {} groups with {} burnin steps".format(ngroups,
                                                                 BURNIN_STEPS))

    # Set up stars' log overlaps with background
    use_background = False
    if bg_ln_ols is not None:
        use_background = True
    elif bg_hist_file:
        logging.info("CORRECTION FACTOR: {}".format(correction_factor))
        use_background = True
        # bg_hists = np.load(rdir + bg_hist_file)
        bg_hists = np.load(bg_hist_file)
        bg_ln_ols = backgroundLogOverlaps(
            star_pars['xyzuvw'], bg_hists,
            correction_factor=correction_factor,
        )
    elif bg_dens:
        logging.info("CORRECTION FACTOR: {}".format(correction_factor))
        use_background = True
        bg_ln_ols = correction_factor * np.log(np.array(nstars * [bg_dens]))

    # INITIALISE GROUPS
    skip_first_e_step = False
    # use init groups if given (along with any z)
    if init_groups is not None:
        z = init_z
    # if just init_z provided, skip first E-step, and maximise off of init_z
    elif init_z is not None:
        skip_first_e_step = True
        # still need a sensible location to begin the walkers
        init_groups = getInitialGroups(ngroups, star_pars['xyzuvw'],
                                       offset=offset)
    # if a synth fit, could initialse at origins
    elif origins is not None:
        init_groups = origins
        z = np.zeros((nstars, ngroups + use_background)) # extra column for bg
        cnt = 0
        for i in range(ngroups):
            z[cnt:cnt+origins[i].nstars, i] = 1.0
            cnt += origins[i].nstars
        logging.info("Initialising fit with origins and membership\n{}".
                     format(z))
    # otherwise, begin with a blind guess
    else:
        init_groups = getInitialGroups(ngroups, star_pars['xyzuvw'],
                                       offset=offset)
        # having z = None triggers an equal weighting of groups in
        # expectation step
        z = None

    np.save(rdir + "init_groups.npy", init_groups)

    old_groups = init_groups
    # old_samples = None
    all_init_pars = [init_group.getInternalSphericalPars() for init_group
                     in init_groups]
    old_overallLnLike = -np.inf
    all_init_pos = ngroups * [None]
    iter_count = 0
    converged = False
    stable_state = True         # used to track issues
    while not converged and stable_state and iter_count < 50:
        # for iter_count in range(10):
        idir = rdir+"iter{:02}/".format(iter_count)
        logging.info("\n--------------------------------------------------"
                     "\n--------------    Iteration {}    ----------------"
                     "\n--------------------------------------------------".
                     format(iter_count))

        mkpath(idir)

        # EXPECTATION
        if skip_first_e_step:
            logging.info("Using input z for first iteration")
            logging.info("z: {}".format(init_z.sum(axis=0)))
            z = init_z
            skip_first_e_step = False
        else:
            z = expectation(star_pars, old_groups, z, bg_ln_ols,
                            inc_posterior=inc_posterior)
            logging.info("Membership distribution:\n{}".format(
                z.sum(axis=0)
            ))
        np.save(idir+"membership.npy", z)

        # MAXIMISE
        #  use `success_mask` to account for groups skipped due to brokenness
        new_groups, all_samples, all_lnprob, all_init_pos, success_mask =\
            maximisation(star_pars, ngroups=ngroups,
                         burnin_steps=BURNIN_STEPS,
                         plot_it=True, pool=pool, convergence_tol=C_TOL,
                         z=z, idir=idir, all_init_pars=all_init_pars,
                         all_init_pos=all_init_pos,
                         ignore_dead_comps=ignore_dead_comps,
                         )

        logging.info("The following groups survived: {}".format(success_mask))

        # apply success mask to z, somewhat awkward cause need to preserve
        # final column (for background overlaps) if present
        if use_background:
            z = np.hstack((z[:,success_mask], z[:,-1][:,np.newaxis]))
        else:
            z = z[:,success_mask]

        # LOG RESULTS OF ITERATION
        overallLnLike = getOverallLnLikelihood(star_pars,
                                               new_groups,
                                               bg_ln_ols, inc_posterior=False)
        # TODO This seems to be bugged... returns same value as lnlike when only
        # fitting one group; BECAUSE WEIGHTS ARE REBALANCED
        overallLnPosterior = getOverallLnLikelihood(star_pars,
                                                    new_groups,
                                                    bg_ln_ols,
                                                    inc_posterior=True)
        logging.info("---        Iteration results         --")
        logging.info("-- Overall likelihood so far: {} --".\
                     format(overallLnLike))
        logging.info("-- Overall posterior so far:  {} --". \
                     format(overallLnPosterior))
        logging.info("-- BIC so far: {}                --".\
                     format(calcBIC(star_pars, ngroups, overallLnLike)))

        # checks if the fit ever worsens
        converged = ( (old_overallLnLike > overallLnLike) and\
                     checkConvergence(old_best_fits=old_groups[success_mask],
                                      new_chains=all_samples,
                                      ))
        # old_samples = all_samples
        old_overallLnLike = overallLnLike
        logging.info("-- Convergence status: {}        --".\
                     format(converged))
        logging.info("---------------------------------------")

        # Ensure stability after sufficient iterations to settle
        if iter_count > 10:
            stable_state = checkStability(star_pars, new_groups, z, bg_ln_ols)

        # only update if the fit has improved
        if not converged:
            # old_old_groups = old_groups
            old_groups = new_groups

        iter_count += 1

    logging.info("CONVERGENCE COMPLETE")
    logging.info("********** EM Algorithm finished *************")

    # TODO: HAVE A THINK ABOUT WHAT RESULTS END UP WHERE...
#    #np.save(rdir+"final_groups.npy", new_groups)
#    np.save(rdir+"final_groups.npy", new_groups) # old grps overwritten by new grps
#    np.save(rdir+"memberships.npy", z)

    if stable_state:
        # PERFORM FINAL EXPLORATION OF PARAMETER SPACE
        logging.info("\n--------------------------------------------------"
                     "\n--------------   Characterising   ----------------"
                     "\n--------------------------------------------------")
        final_dir = rdir+"final/"
        mkpath(final_dir)

        final_z = expectation(star_pars, new_groups, z, bg_ln_ols,
                              inc_posterior=inc_posterior)
        np.save(final_dir+"final_membership.npy", final_z)
        final_best_fits = [None] * ngroups
        final_med_errs = [None] * ngroups

        for i in range(ngroups):
            logging.info("Characterising group {}".format(i))
            final_gdir = final_dir + "group{}/".format(i)
            mkpath(final_gdir)

            best_fit, chain, lnprob = gf.fitGroup(
                xyzuvw_dict=star_pars, burnin_steps=BURNIN_STEPS,
                plot_it=True, pool=pool, convergence_tol=C_TOL,
                plot_dir=final_gdir, save_dir=final_gdir, z=z[:, i],
                init_pos=all_init_pos[i], sampling_steps=SAMPLING_STEPS,
                max_iter=4
                # init_pars=old_groups[i],
            )
            # run with extremely large convergence tolerance to ensure it only
            # runs once
            logging.info("Finished fit")
            final_best_fits[i] = best_fit
            final_med_errs[i] = calcMedAndSpan(chain)
            # np.save(final_gdir + "best_group_fit.npy", new_group)
            np.save(final_gdir + 'final_chain.npy', chain)
            np.save(final_gdir + 'final_lnprob.npy', lnprob)

            all_init_pos[i] = chain[:, -1, :]


        final_groups = np.array(
            [syn.Group(final_best_fit, sphere=True, internal=True,
                       starcount=False)
             for final_best_fit in final_best_fits]
        )
        np.save(final_dir+'final_groups.npy', final_groups)
        np.save(final_dir+'final_med_errs.npy', final_med_errs)

        # get overall likelihood
        overallLnLike = getOverallLnLikelihood(star_pars, new_groups,
                                               bg_ln_ols, inc_posterior=False)
        overallLnPosterior = getOverallLnLikelihood(star_pars, new_groups,
                                                bg_ln_ols, inc_posterior=True)
        bic = calcBIC(star_pars, ngroups, overallLnLike)
        logging.info("Final overall lnlikelihood: {}".format(overallLnLike))
        logging.info("Final overall lnposterior:  {}".format(overallLnLike))
        logging.info("Final BIC: {}".format(bic))

        np.save(final_dir+'likelihood_post_and_bic.npy', (overallLnLike,
                                                          overallLnPosterior,
                                                          bic))

        logging.info("FINISHED CHARACTERISATION")
        #logging.info("Origin:\n{}".format(origins))
        logging.info("Best fits:\n{}".format(
            [fg.getSphericalPars() for fg in final_groups]
        ))
        logging.info("Stars per component:\n{}".format(z.sum(axis=0)))
        logging.info("Memberships: \n{}".format((z*100).astype(np.int)))

        return final_groups, np.array(final_med_errs), z

    else: # not stable_state
        logging.info("****************************************")
        logging.info("********** BAD RUN TERMINATED **********")
        logging.info("****************************************")
        return new_groups, -1, z

