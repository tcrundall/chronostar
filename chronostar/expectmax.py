"""
a module for implementing the expectation-maximisation algorithm
in order to fit a multi-gaussian mixture model of moving groups' origins
to a data set of stars tracedback through XYZUVW

todo:
    - implement average error cacluation in lnprobfunc
"""
from __future__ import print_function, division

import pdb
from distutils.dir_util import mkpath
import logging
import numpy as np
import os

try:
    import matplotlib as mpl
    # prevents displaying plots from generation from tasks in background
    mpl.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not imported")

from chronostar.component import SphereComponent
from chronostar import likelihood
from . import groupfitter
from . import tabletool


def log_message(msg, symbol='.', surround=False):
    """Little formatting helper"""
    res = '{}{:^40}{}'.format(5*symbol, msg, 5*symbol)
    if surround:
        res = '\n{}\n{}\n{}'.format(50*symbol, res, 50*symbol)
    logging.info(res)


def checkConvergence(old_best_comps, new_chains,
                     perc=40):
    """Check if the last maximisation step yielded is consistent to new fit

    Note, percentage raised to 40 (from 25) as facing issues establishing
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
    # import pdb; pdb.set_trace()

    for old_best_comp, new_chain in zip(old_best_comps, new_chains):
        errors = groupfitter.calc_med_and_span(new_chain, perc=perc)
        upper_contained =\
            old_best_comp.internalise(old_best_comp.get_pars()) < errors[:, 1]
        lower_contained =\
            old_best_comp.internalise(old_best_comp.get_pars()) > errors[:, 2]

        each_converged.append(
            np.all(upper_contained) and np.all(lower_contained))

    return np.all(each_converged)


def calcMembershipProbs(star_lnols):
    """Calculate probabilities of membership for a single star from overlaps

    Parameters
    ----------
    star_lnols : [ncomps] array
        The log of the overlap of a star with each group

    Returns
    -------
    star_memb_probs : [ncomps] array
        The probability of membership to each group, normalised to sum to 1
    """
    ncomps = star_lnols.shape[0]
    star_memb_probs = np.zeros(ncomps)

    for i in range(ncomps):
        star_memb_probs[i] = 1. / np.sum(np.exp(star_lnols - star_lnols[i]))

    return star_memb_probs


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


def getAllLnOverlaps(data, comps, old_memb_probs=None, bg_ln_ols=None,
                     inc_posterior=False, amp_prior=None):
    """
    Get the log overlap integrals of each star with each component

    Parameters
    ----------
    data : dict
        stars: (nstars) high astropy table including columns as
                    documented in the Traceback class.
        times : [ntimes] numpy array
            times that have been traced back, in Myr
        xyzuvw : [nstars, ntimes, 6] array
            XYZ in pc and UVW in km/s
        xyzuvw_cov : [nstars, ntimes, 6, 6] array
            covariance of xyzuvw

    comps : [ncomps] syn.Group object list
        a fit for each comp (in internal form)

    old_memb_probs : [nstars, ncomps (+1)] float array {None}
        Only used to get weights (amplitudes) for each fitted component.
        Tracks membership probabilities of each star to each comp. Each
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
    lnols: [nstars, ncomps (+1)] float array
        The log overlaps of each star with each component, optionally
        with the log background overlaps appended as the final column
    """
    if not isinstance(data, dict):
        data = tabletool.buildDataFromTable(data)
    nstars = len(data['means'])
    ncomps = len(comps)
    using_bg = bg_ln_ols is not None

    lnols = np.zeros((nstars, ncomps + using_bg))

    if old_memb_probs is None:
        old_memb_probs = np.ones((nstars, ncomps)) / ncomps


    weights = old_memb_probs[:, :ncomps].sum(axis=0)

    # Optionally scale each weight by the component prior, then rebalance
    # so total expected stars across all components is unchanged
    if inc_posterior:
        comp_lnpriors = np.zeros(ncomps)# &TC
        for i, comp in enumerate(comps):
            comp_lnpriors[i] = likelihood.ln_alpha_prior(
                    comp, memb_probs=old_memb_probs
            )
        ngroup_stars = weights.sum()
        weights *= np.exp(comp_lnpriors)
        weights = weights / weights.sum() * ngroup_stars

    # Optionally scale each weight such that the total expected stars
    # is equal to or greater than `amp_prior`
    if amp_prior:
        if weights.sum() < amp_prior:
            weights *= amp_prior / weights.sum()

#    # except:
#    logging.info("_____ DEBUGGGING _____")
#    logging.info("ncomps: {}".format(ncomps))
#    logging.info("old_z shape: {}".format(old_z.shape))
#    logging.info("weights shape: {}".format(weights.shape))
#    logging.info("weights: {}".format(weights))
#    import pdb; pdb.set_trace()

    for i, comp in enumerate(comps):
        # weight is the amplitude of a component, proportional to its expected
        # total of stellar members
        #weight = old_z[:,i].sum()
        # threshold = nstars/(2. * (ncomps+1))
        # if weight < threshold:
        #     logging.info("!!! GROUP {} HAS LESS THAN {} STARS, weight: {}".\
        #         format(i, threshold, weight)
        # )
        # comp_pars = comp.internalise(comp.get_pars())
        lnols[:, i] = \
            np.log(weights[i]) + \
            likelihood.get_lnoverlaps(comp, data)

    # insert one time calculated background overlaps
    if using_bg:
        lnols[:,-1] = bg_ln_ols
    return lnols


def calcBIC(data, ncomps, lnlike, z=None):
    """Calculates the Bayesian Information Criterion

    A simple metric to judge whether added components are worthwhile

    Could just as legitimately have n = to number of reasonably likely
    members to provided components
    Currenty trialing this:
    """
    if z is not None:
        nstars = np.sum(z[:,:ncomps])
    else:
        nstars = len(data)
    n = nstars * 6 # 6 for phase space origin (and 1 for age???)
    k = ncomps * 8 # 6 for central estimate, 2 for dx and dv, 1 for age
    return np.log(n)*k - 2 * lnlike


def expectation(data, comps, old_memb_probs=None, bg_ln_ols=None,
                inc_posterior=False, amp_prior=None):
    """Calculate membership probabilities given fits to each group

    Parameters
    ----------
    data : dict
        stars: (nstars) high astropy table including columns as
                    documented in the Traceback class.
        times : [ntimes] numpy array
            times that have been traced back, in Myr
        xyzuvw : [nstars, ntimes, 6] array
            XYZ in pc and UVW in km/s
        xyzuvw_cov : [nstars, ntimes, 6, 6] array
            covariance of xyzuvw

    comps : [ncomps] syn.Group object list
        a fit for each group (in internal form)

    old_memb_probs : [nstars, ncomps (+1)] float array
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
    memb_probs : [nstars, ncomps] array
        An array designating each star's probability of being a member to
        each group. It is populated by floats in the range (0.0, 1.0) such
        that each row sums to 1.0, each column sums to the expected size of
        each group, and the entire array sums to the number of stars.
    """
    if not isinstance(data, dict):
        data = tabletool.buildDataFromTable(data)
    # import pdb; pdb.set_trace()
    ncomps = len(comps)
    nstars = len(data['means'])

    using_bg = bg_ln_ols is not None

    # if no memb_probs provided, assume perfectly equal membership
    if old_memb_probs is None:
        old_memb_probs = np.ones((nstars, ncomps+using_bg)) / (ncomps+using_bg)

    lnols = getAllLnOverlaps(data, comps, old_memb_probs, bg_ln_ols,
                             inc_posterior=inc_posterior, amp_prior=amp_prior)

    memb_probs = np.zeros((nstars, ncomps + using_bg))
    for i in range(nstars):
        memb_probs[i] = calcMembershipProbs(lnols[i])
    if np.isnan(memb_probs).any():
        logging.info("!!!!!! AT LEAST ONE MEMBERSHIP IS 'NAN' !!!!!!")
        memb_probs[np.where(np.isnan(memb_probs))] = 0.
        # import pdb; pdb.set_trace()
    return memb_probs


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


def getInitialGroups(ncomps, xyzuvw, offset=False, v_dist=10.,
                     Component=SphereComponent):
    """
    Generate the parameter list with which walkers will be initialised

    TODO: replace hardcoding parameter generation with Component methods

    Parameters
    ----------
    ncomps: int
        number of comps
    xyzuvw: [nstars, 6] array
        the mean measurement of stars
    offset : (boolean {False})
        If set, the gorups are initialised in the complementary angular
        positions
    v_dist: float
        Radius of circle in UV plane along which comps are initialsed

    Returns
    -------
    comps: [ngroups] synthesiser.Group object list
        the parameters with which to initialise each comp's emcee run
    """
    comps = []

    # if only fitting one comp, simply initialise walkers about the
    # mean of the data set
    if ncomps == 1:
        v_dist = 0

    mean = np.mean(xyzuvw, axis=0)[:6]
    logging.info("Mean is\n{}".format(mean))
#    meanXYZ = np.array([0.,0.,0.])
#    meanW = 0.
    dx = 100.
    dv = 15.
    age = 0.5
    # group_pars_base = list([0, 0, 0, None, None, 0, np.log(50),
    #                         np.log(5), 3])
    pts = getPointsOnCircle(npoints=ncomps, v_dist=v_dist, offset=offset)
    logging.info("Points around circle are:\n{}".format(pts))

    for i in range(ncomps):
        mean_w_offset = np.copy(mean)
        mean_w_offset[3:5] += pts[i]
        logging.info("Group {} has init UV of ({},{})".\
                    format(i, mean_w_offset[3], mean_w_offset[4]))
        comp_pars = np.hstack((mean_w_offset, dx, dv, age))
        comp = Component(comp_pars)
        comps.append(comp)

    return np.array(comps)
#
# def decomposeGroup(comp, young_age=None, old_age=None, age_offset=4):
#     """
#     Takes a group object and splits it into two components offset by age.
#
#     Parameters
#     ----------
#     comp: synthesiser.Group instance
#         the group which is to be decomposed
#
#     Returns
#     -------
#     all_init_pars: [2, npars] array
#         the intenralised parameters with which the walkers will be
#         initiallised
#     sub_groups: [2] list of Group instances
#         the group objects of the resulting decomposition
#     """
#     internal_pars = comp.getInternalSphericalPars()
#     mean_now = torb.traceOrbitXYZUVW(comp.mean, comp.age, single_age=True)
#     ngroups = 2
#
#     sub_groups = []
#
#     if (young_age is None) and (old_age is None):
#         young_age = max(1e-5, comp.age - age_offset)
#         old_age = comp.age + age_offset
#
#     ages = [young_age, old_age]
#     for age in ages:
#         mean_then = torb.traceOrbitXYZUVW(mean_now, -age, single_age=True)
#         group_pars_int = np.hstack((mean_then, internal_pars[6:8], age))
#         sub_groups.append(
#             component.Component(group_pars_int, form='sphere',
#                                            internal=True))
#     all_init_pars = [sg.getInternalSphericalPars() for sg in sub_groups]
#
#     return all_init_pars, sub_groups


def getOverallLnLikelihood(data, comps, bg_ln_ols=None, return_z=False,
                           inc_posterior=False):
    """
    Get overall likelihood for a proposed model.

    Evaluates each star's overlap with every component and background
    If only fitting one group, inc_posterior does nothing

    Parameters
    ----------
    data : (dict)
    comps : [ngroups] list of Synthesiser.Group objects
    memb_probs : [nstars, ngroups] float array
        membership array
    bg_ln_ols : [nstars] float array
        the overlap each star has with the provided background density
        distribution

    Returns
    -------
    overall_lnlikelihood : float
    """
    memb_probs = expectation(data, comps, None, bg_ln_ols,
                             inc_posterior=inc_posterior)
    all_ln_ols = getAllLnOverlaps(data, comps, memb_probs, bg_ln_ols,
                                  inc_posterior=inc_posterior)

    # multiplies each log overlap by the star's membership probability
    # import pdb; pdb.set_trace()
    weighted_lnols = np.einsum('ij,ij->ij', all_ln_ols, memb_probs)
    if return_z:
        return np.sum(weighted_lnols), memb_probs
    else:
        return np.sum(weighted_lnols)


def maximisation(data, ncomps, memb_probs, burnin_steps, idir,
                 all_init_pars, all_init_pos=None, plot_it=False, pool=None,
                 convergence_tol=0.25, ignore_dead_comps=False,
                 Component=SphereComponent,
                 trace_orbit_func=None):
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
    :param memb_probs:
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

    new_comps = []
    all_samples = []
    all_lnprob = []
    success_mask = []
    if all_init_pos is None:
        all_init_pos = ncomps * [None]

    for i in range(ncomps):
        log_message('Fitting comp {}'.format(i), symbol='.', surround=True)
        # logging.info("........................................")
        # logging.info("          Fitting comp {}".format(i))
        # logging.info("........................................")
        gdir = idir + "comp{}/".format(i)
        mkpath(gdir)
        # pdb.set_trace()
        # If component has too few stars, skip fit, and use previous best walker
        if ignore_dead_comps and (np.sum(memb_probs[:, i]) < DEATH_THRESHOLD):
            logging.info("Skipped component {} with nstars {}".format(
                    i, np.sum(memb_probs[:, i])
            ))
        else:
            best_comp, chain, lnprob = groupfitter.fit_comp(
                    data=data,
                    memb_probs=memb_probs[:, i],
                    burnin_steps=burnin_steps,
                    plot_it=plot_it,
                    pool=pool,
                    convergence_tol=convergence_tol,
                    plot_dir=gdir,
                    save_dir=gdir,
                    init_pos=all_init_pos[i],
                    init_pars=all_init_pars[i],
                    Component=Component,
                    trace_orbit_func=trace_orbit_func,
            )
            logging.info("Finished fit")
            logging.info("Best comp pars:\n{}".format(
                    best_comp.get_pars()
            ))

            final_pos = chain[:, -1, :]

            logging.info("With age of: {:.3} +- {:.3} Myr".\
                        format(np.median(chain[:,:,-1]),
                               np.std(chain[:,:,-1])))
            # new_comp = Component(best_comp, internal=True)
            new_comps.append(best_comp)
            np.save(gdir + "best_comp_fit.npy", best_comp)
            np.save(gdir + 'final_chain.npy', chain)
            np.save(gdir + 'final_lnprob.npy', lnprob)
            all_samples.append(chain)
            all_lnprob.append(lnprob)

            success_mask.append(i)

            # record the final position of the walkers for each comp
            # TODO: TIM TO FIX IMPENDING BUG HERE
            all_init_pos[i] = final_pos

    np.save(idir + 'best_comps.npy', new_comps)

    return np.array(new_comps), np.array(all_samples), np.array(all_lnprob),\
           np.array(all_init_pos), np.array(success_mask)


def checkStability(star_pars, best_comps, z, bg_ln_ols=None):
    """
    Checks if run has encountered problems

    Common problems include: a component losing all its members, lnprob
    return nans, a membership listed as nan

    Paramters
    ---------
    star_pars : dict
        Contains XYZUVW kinematics of stars
    best_groups : [ncomps] list of Synthesiser.Group objects
        The best fits (np.argmax(chain)) for each component from the most
        recent run
    z : [nstars, ncomps] float array
        The membership array from the most recent run
    bg_ln_ols : [nstars] float array
        The overlap the stars have with the (fixed) background distribution
    """
    ncomps = len(best_comps)
    stable = True
    if np.min(np.sum(z[:,:ncomps], axis=0)) <= 2.:
        logging.info("ERROR: A component has less than 2 members")
        stable = False
    if not np.isfinite(getOverallLnLikelihood(star_pars,
                                              best_comps,
                                              bg_ln_ols)):
        logging.info("ERROR: Posterior is not finite")
        stable = False
    if not np.isfinite(z).all():
        logging.info("ERROR: At least one membership is not finite")
        stable = False
    return stable


def fitManyGroups(data, ncomps, rdir='', init_memb_probs=None,
                  origins=None, pool=None, init_with_origin=False,
                  init_comps=None, init_weights=None,
                  offset=False, bg_hist_file='', correction_factor=1.0,
                  inc_posterior=False, burnin=1000, bg_dens=None,
                  bg_ln_ols=None, ignore_dead_comps=False,
                  Component=SphereComponent,
                  trace_orbit_func=None):
    """
    Entry point: Fit multiple Gaussians to data set

    Parameters
    ----------
    data: dict
        'xyzuvw': [nstars, 6] numpy array
            the xyzuvw mean values of each star, calculated from astrometry
        'xyzuvw_cov': [nstars, 6, 6] numpy array
            the xyzuvw covarince values of each star, calculated from
            astrometry
        'table': Astropy table (sometimes None)
            The astrometry from which xyzuvw values are calculated. Can
            optionally include more information, like star names etc.
    ncomps: int
        the number of groups to be fitted to the data
    rdir: String {''}
        The directory in which all the data will be stored and accessed
        from
    init_memb_probs: [nstars, ngroups] array {None} [UNIMPLEMENTED]
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
    final_comps: [ngroups] list of synthesiser.Group objects
        the best fit for each component
    final_med_errs: [ngroups, npars, 3] array
        the median, -34 perc, +34 perc values of each parameter from
        each final sampling chain
    memb_probs: [nstars, ngroups] array
        membership probabilities

    TODO: Generalise interventions for more than 2 groups
    TODO: Allow option with which step to start with
    """
    # Tidying up input
    if not isinstance(data, dict):
        data = tabletool.buildDataFromTable(data)
    if rdir == '':                      # Ensure results directory has a
        rdir = '.'                      # trailing '/'
    rdir = rdir.rstrip('/') + '/'
    if not os.path.exists(rdir):
        mkpath(rdir)

    # setting up some constants
    BURNIN_STEPS = burnin
    SAMPLING_STEPS = 5000
    C_TOL = 0.5
    MAX_ITERS = 100
    # MEMB_CONV_TOL = 0.1 # no memberships may vary by >10% to be converged.
    AMPLITUDE_TOL = 1.0 # total sum of memberships for each component
                        # cannot vary by more than this value to be converged
    nstars = data['means'].shape[0]

    logging.info("Fitting {} groups with {} burnin steps".format(ncomps,
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
                tabletool.buildDataFromTable(data, only_means=True),
                bg_hists,
                correction_factor=correction_factor,
        )
    elif bg_dens:
        logging.info("CORRECTION FACTOR: {}".format(correction_factor))
        use_background = True
        bg_ln_ols = correction_factor * np.log(np.array(nstars * [bg_dens]))

    # INITIALISE GROUPS
    skip_first_e_step = False
    # use init groups if given (along with any memb_probs)
    if init_comps is not None:
        memb_probs_old = init_memb_probs
    # if just init_z provided, skip first E-step, and maximise off of init_z
    elif init_memb_probs is not None:
        skip_first_e_step = True
        # still need a sensible location to begin the walkers
        init_comps = getInitialGroups(
                ncomps,
                tabletool.buildDataFromTable(data, only_means=True),
                offset=offset
        )
    # if a synth fit, could initialse at origins
    elif origins is not None:
        init_comps = origins
        memb_probs_old = np.zeros((nstars, ncomps + use_background)) # extra column for bg
        cnt = 0
        for i in range(ncomps):
            memb_probs_old[cnt:cnt+origins[i].nstars, i] = 1.0
            cnt += origins[i].nstars
        logging.info("Initialising fit with origins and membership\n{}".
                     format(memb_probs_old))
    # otherwise, begin with a blind guess
    else:
        init_comps = getInitialGroups(
                ncomps,
                data['means'],
                offset=offset,
        )
        # having memb_probs = None triggers an equal weighting of groups in
        # expectation step
        # TODO: Handle this more smoothly... don't leave things None for hidden reasons
        memb_probs_old = np.ones((nstars, ncomps+use_background))/\
                         (ncomps+use_background)
        # memb_probs_old = None

    np.save(rdir + "init_groups.npy", init_comps)

    # Initialise values for upcoming iterations
    old_comps = init_comps
    all_init_pars = [Component.internalise(init_comp.get_pars()) for
                     init_comp in init_comps]
    old_overallLnLike = -np.inf
    all_init_pos = ncomps * [None]
    iter_count = 0
    all_converged = False
    stable_state = True         # used to track issues

    # Look for previous iterations and overwrite values as appropriate
    prev_iters = True
    iter_count = 0
    while prev_iters:
        try:
            idir = rdir+"iter{:02}/".format(iter_count)
            # old_comps = np.load(idir + 'best_comps.npy')
            old_comps = Component.load_components(idir + 'best_comps.npy')
            memb_probs_old = np.load(idir + 'membership.npy')
            old_overallLnLike = getOverallLnLikelihood(data, old_comps,
                                                       bg_ln_ols,
                                                       inc_posterior=False)
            all_init_pars = [Component.internalise(old_comp.get_pars())
                             for old_comp in old_comps]
            iter_count += 1
        except IOError:
            logging.info("Managed to find {} previous iterations".format(
                iter_count
            ))
            prev_iters = False


    while not all_converged and stable_state and iter_count < MAX_ITERS:
        # for iter_count in range(10):
        idir = rdir+"iter{:02}/".format(iter_count)
        log_message('Iteration {}'.format(iter_count),
                    symbol='-', surround=True)

        mkpath(idir)

        # EXPECTATION
        if skip_first_e_step:
            logging.info("Using input memb_probs for first iteration")
            logging.info("memb_probs: {}".format(init_memb_probs.sum(axis=0)))
            memb_probs_new = init_memb_probs
            skip_first_e_step = False
        else:
            memb_probs_new = expectation(data, old_comps, memb_probs_old, bg_ln_ols,
                                inc_posterior=inc_posterior)
            logging.info("Membership distribution:\n{}".format(
                memb_probs_new.sum(axis=0)
            ))
        np.save(idir+"membership.npy", memb_probs_new)

        # MAXIMISE
        #  use `success_mask` to account for groups skipped due to brokenness
        new_comps, all_samples, all_lnprob, all_init_pos, success_mask =\
            maximisation(data, ncomps=ncomps,
                         burnin_steps=BURNIN_STEPS,
                         plot_it=True, pool=pool, convergence_tol=C_TOL,
                         memb_probs=memb_probs_new, idir=idir,
                         all_init_pars=all_init_pars,
                         all_init_pos=all_init_pos,
                         ignore_dead_comps=ignore_dead_comps,
                         trace_orbit_func=trace_orbit_func,
                         )

        # update number of groups to reflect any loss of dead components
        ncomps = len(success_mask)
        logging.info("The following components survived: {}".format(
                success_mask
        ))

        # apply success mask to memb_probs, somewhat awkward cause need to preserve
        # final column (for background overlaps) if present
        if use_background:
            memb_probs_new = np.hstack((memb_probs_new[:,success_mask],
                                    memb_probs_new[:,-1][:,np.newaxis]))
        else:
            memb_probs_new = memb_probs_new[:,success_mask]

        # LOG RESULTS OF ITERATION
        overallLnLike = getOverallLnLikelihood(data,
                                               new_comps,
                                               bg_ln_ols, inc_posterior=False)
        # TODO This seems to be bugged... returns same value as lnlike when only
        # fitting one group; BECAUSE WEIGHTS ARE REBALANCED
        overallLnPosterior = getOverallLnLikelihood(data,
                                                    new_comps,
                                                    bg_ln_ols,
                                                    inc_posterior=True)
        logging.info("---        Iteration results         --")
        logging.info("-- Overall likelihood so far: {} --".\
                     format(overallLnLike))
        logging.info("-- Overall posterior so far:  {} --". \
                     format(overallLnPosterior))
        logging.info("-- BIC so far: {}                --". \
                     format(calcBIC(data, ncomps, overallLnLike,
                                    z=memb_probs_new)))

        # checks if the fit ever worsens
        # import pdb; pdb.set_trace()
        chains_converged = checkConvergence(
                old_best_comps=np.array(old_comps)[success_mask],
                new_chains=all_samples
        )
        amplitudes_converged = np.allclose(memb_probs_new.sum(axis=0),
                                           memb_probs_old.sum(axis=0),
                                           atol=AMPLITUDE_TOL)
        likelihoods_converged = (old_overallLnLike > overallLnLike)
        all_converged = (chains_converged and amplitudes_converged and
                         likelihoods_converged)
        # old_samples = all_samples
        old_overallLnLike = overallLnLike
        log_message('Convergence status: {}'.format(all_converged),
                    symbol='-', surround=True)
        if not all_converged:
            logging.info('Likelihoods converged: {}'. \
                         format(likelihoods_converged))
            logging.info('Chains converged: {}'.format(chains_converged))
            logging.info('Amplitudes converged: {}'.\
                format(amplitudes_converged))


        # Ensure stability after sufficient iterations to settle
        if iter_count > 10:
            stable_state = checkStability(data, new_comps, memb_probs_new, bg_ln_ols)

        # only update if the fit has improved
        if not all_converged:
            # old_old_groups = old_comps
            old_comps = new_comps
            memb_probs_old = memb_probs_new

        iter_count += 1

    logging.info("CONVERGENCE COMPLETE")
    log_message('EM Algorithm finished', symbol='*')


    if stable_state:
        # PERFORM FINAL EXPLORATION OF PARAMETER SPACE
        log_message('Characterising', symbol='-', surround=True)
        final_dir = rdir+"final/"
        mkpath(final_dir)

        memb_probs_final = expectation(data, new_comps, memb_probs_new,
                                       bg_ln_ols, inc_posterior=inc_posterior)
        np.save(final_dir+"final_membership.npy", memb_probs_final)
        final_med_and_spans = [None] * ncomps
        final_best_comps = [None] * ncomps

        for i in range(ncomps):
            logging.info("Characterising comp {}".format(i))
            final_gdir = final_dir + "comp{}/".format(i)
            mkpath(final_gdir)

            best_comp, chain, lnprob = groupfitter.fit_comp(
                    data=data,
                    memb_probs=memb_probs_final[:, i],
                    burnin_steps=BURNIN_STEPS,
                    plot_it=True, pool=pool, convergence_tol=C_TOL,
                    plot_dir=final_gdir, save_dir=final_gdir,
                    init_pos=all_init_pos[i],
                    sampling_steps=SAMPLING_STEPS,
                    trace_orbit_func=trace_orbit_func,
                # max_iter=4 # Todo: why max_iter? (19/02)
                # init_pars=old_comps[i],
            )
            # run with extremely large convergence tolerance to ensure it only
            # runs once
            logging.info("Finished fit")
            final_best_comps[i] = best_comp
            final_med_and_spans[i] = groupfitter.calc_med_and_span(
                    chain, intern_to_extern=True, Component=Component,
            )
            # np.save(final_gdir + "best_group_fit.npy", new_group)
            np.save(final_gdir + 'final_chain.npy', chain)
            np.save(final_gdir + 'final_lnprob.npy', lnprob)

            all_init_pos[i] = chain[:, -1, :]


        # final_comps = np.array(
        #     [Component(pars=final_best_fit, internal=True)
        #      for final_best_fit in final_best_fits]
        # )
        np.save(final_dir+'final_comps.npy', final_best_comps)
        np.save(final_dir+'final_med_and_spans.npy', final_med_and_spans)

        # get overall likelihood
        overallLnLike = getOverallLnLikelihood(
                data, new_comps, bg_ln_ols, inc_posterior=False
        )
        overallLnPosterior = getOverallLnLikelihood(
                data, new_comps, bg_ln_ols, inc_posterior=True
        )
        bic = calcBIC(data, ncomps, overallLnLike, z=memb_probs_final)
        logging.info("Final overall lnlikelihood: {}".format(overallLnLike))
        logging.info("Final overall lnposterior:  {}".format(overallLnLike))
        logging.info("Final BIC: {}".format(bic))

        np.save(final_dir+'likelihood_post_and_bic.npy', (overallLnLike,
                                                          overallLnPosterior,
                                                          bic))

        logging.info("FINISHED CHARACTERISATION")
        #logging.info("Origin:\n{}".format(origins))
        logging.info("Best fits:\n{}".format(
            [fc.get_pars() for fc in final_best_comps]
        ))
        logging.info("Stars per component:\n{}".format(
                memb_probs_final.sum(axis=0)
        ))
        logging.info("Memberships: \n{}".format(
                (memb_probs_final*100).astype(np.int)
        ))

        return final_best_comps, np.array(final_med_and_spans), memb_probs_final

    else: # not stable_state
        log_message('BAD RUN TERMINATED', symbol='*', surround=True)
        # logging.info("****************************************")
        # logging.info("********** BAD RUN TERMINATED **********")
        # logging.info("****************************************")
        return new_comps, -1, memb_probs_new

