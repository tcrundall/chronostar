"""
a module for implementing the expectation-maximisation algorithm
in order to fit a multi-gaussian mixture model of moving groups' origins
to a data set of stars tracedback through XYZUVW

todo:
    - implement average error cacluation in lnprobfunc
"""
from __future__ import print_function, division

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
    if array is None:
        return None
    else:
        return array[ix]


def ix_snd(array, ix):
    if array is None:
        return None
    else:
        return array[:, ix]


def calcErrors(chain, perc=34):
    """
    Given a set of aligned (converted?) samples, calculate the median and
    errors of each parameter

    Parameters
    ----------
    chain : [nwalkers, nsteps, npars]
        The chain of samples (in internal encoding)
    """
    npars = chain.shape[-1]  # will now also work on flatchain as input
    flat_chain = np.reshape(chain, (-1, npars))

    #    conv_chain = np.copy(flat_chain)
    #    conv_chain[:, 6:10] = 1/conv_chain[:, 6:10]

    # return np.array( map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
    #                 zip(*np.percentile(flat_chain, [16,50,84], axis=0))))
    return np.array(map(lambda v: (v[1], v[2], v[0]),
                        zip(*np.percentile(flat_chain,
                                           [50-perc, 50, 50+perc],
                                           axis=0))))


def checkConvergence(old_best_fits, new_chains,
                     perc=25):
    """Check if the last maximisation step yielded is consistent to new fit

    TODO: incorporate Z into this convergence checking. e.g.
    np.allclose(z_prev, z, rtol=1e-2)

    Convergence is achieved if previous key values fall within +/-"perc" of
    the new fits.

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
        errors = calcErrors(new_chain, perc=perc)
        upper_contained =\
            old_best_fit.getInternalSphericalPars() < errors[:, 1]
        lower_contained =\
            old_best_fit.getInternalSphericalPars() > errors[:, 2]

        each_converged.append(
            np.all(upper_contained) and np.all(lower_contained))

    return np.all(each_converged)


def calcLnoverlaps(group_pars, star_pars, nstars):
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
    lnols = None
    return lnols


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


def expectation(star_pars, groups, old_z=None):
    """Calculate membership probabilities given fits to each group
    TODO: incorporate group sizes into the weighting

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

    # if no z provided, assume perfectly equal membership
    if old_z is None:
        old_z = np.ones((nstars, ngroups))/ngroups

    lnols = np.zeros((nstars, ngroups))
    for i, group in enumerate(groups):
        weight = old_z[:,i].sum()
        threshold = nstars/(2. * (ngroups+1))
        if weight < threshold:
            logging.info("!!! GROUP {} HAS LESS THAN {} STARS, weight: {}".\
                format(i, threshold, weight)
        )
        group_pars = group.getInternalSphericalPars()
        lnols[:, i] =\
            np.log(weight) + gf.lnlike(group_pars, star_pars, return_lnols=True)
        # calc_lnoverlaps(group_pars, star_pars, nstars)
    z = np.zeros((nstars, ngroups))
    for i in range(nstars):
        z[i] = calcMembershipProbs(lnols[i])
    if np.isnan(z).any():
        import pdb; pdb.set_trace()
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


def getInitialGroups(ngroups, xyzuvw, offset=False):
    """
    Generate the parameter list with which walkers will be initialised
    TODO: CENTRE THIS BY MEAN U AND V OF STARS

    Parameters
    ----------
    ngroups: int
        number of groups
    xyzuvw: [nstars, 6] array
        the mean measurement of stars
    offset : (boolean {False})
        If set, the gorups are initialised in the complementary angular
        positions

    Returns
    -------
    groups: [ngroups] synthesiser.Group object list
        the parameters with which to initialise each group's emcee run
    """
    groups = []

    mean = np.mean(xyzuvw, axis=0)[:6]
#    meanXYZ = np.array([0.,0.,0.])
#    meanW = 0.
    dx = 50.
    dv = 5.
    age = 3.
    group_pars_base = list([0, 0, 0, None, None, 0, np.log(50),
                            np.log(5), 3])
    pts = getPointsOnCircle(npoints=ngroups, v_dist=10, offset=offset)

    for i in range(ngroups):
        mean[3:5] = mean[3:5] + pts[i]
        group_pars = np.hstack((mean, dx, dv, age))
        group = syn.Group(group_pars, sphere=True, starcount=False)
        groups.append(group)

    return groups

def decomposeGroup(group):
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
    AGE_OFFSET = 4

    sub_groups = []

    young_age = max(0., group.age - AGE_OFFSET)
    old_age = group.age + AGE_OFFSET

    ages = [young_age, old_age]
    for age in ages:
        mean_then = torb.traceOrbitXYZUVW(mean_now, -age, single_age=True)
        group_pars_int = np.hstack(mean_then, internal_pars[6:8], age)
        sub_groups.append(syn.Group(group_pars_int, sphere=True,
                                    internal=True, starcount=False))
    all_init_pars = [sg.getInternalSphericalPars() for sg in sub_groups]

    return all_init_pars, sub_groups


def fitManyGroups(star_pars, ngroups, rdir='', init_z=None,
                  origins=None, pool=None):
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

    Return
    ------
    final_groups: [ngroups, npars] array
        the best fit for each group
    final_med_errs: [ngroups, npars, 3] array
        the median, -34 perc, +34 perc values of each parameter from
        each final sampling chain
    z: [nstars, ngroups] array
        membership probabilities
    """
    # setting up some constants
    BURNIN_STEPS = 1000
    SAMPLING_STEPS = 5000
    C_TOL = 0.5

    # INITIALISE GROUPS
    init_groups = getInitialGroups(ngroups, star_pars['xyzuvw'])

    np.save(rdir + "init_groups.npy", init_groups)


    all_init_pos = ngroups * [None]
    iter_count = 0
    converged = False

    # having z = None triggers an equal weighting of groups in
    # expectation step
    z = None
    old_groups = init_groups
    all_init_pars = [init_group.getInternalSphericalPars() for init_group
                     in init_groups]

    while not converged:
        # for iter_count in range(10):
        idir = rdir+"iter{}/".format(iter_count)
        logging.info("\n--------------------------------------------------"
                     "\n--------------    Iteration {}    ----------------"
                     "\n--------------------------------------------------".
                     format(iter_count))

        mkpath(idir)
        #os.chdir("iter{}".format(iter_count))

        # EXPECTATION
        z = expectation(star_pars, old_groups, z)

        logging.info("Membership distribution:\n{}".format(
            z.sum(axis=0)
        ))
        if (min(z.sum(axis=0)) < 10):
            logging.info("!!! WARNING, GROUP {} HAS LESS THAN 10 STARS".\
                         format(np.argmin(z.sum(axis=0))))
            logging.info("+++++++++++++++++++++++++++++++++++++++++++")
            logging.info("++++            Intervening            ++++")
            logging.info("+++++++++++++++++++++++++++++++++++++++++++")
            all_init_pos = [None] * ngroups
            all_init_pars, sub_groups =\
                decomposeGroup(old_groups[np.argmax(z.sum(axis=0))])
        np.save(idir+"init_subgroups.npy", sub_groups)
        np.save(idir+"membership.npy", z)

        # MAXIMISE
        #new_groups = np.zeros(old_groups.shape)
        new_groups = []

        all_samples = []
        all_lnprob = []

        for i in range(ngroups):
            logging.info("........................................")
            logging.info("          Fitting group {}".format(i))
            logging.info("........................................")
            gdir = idir + "group{}/".format(i)
            mkpath(gdir)

            best_fit, chain, lnprob = gf.fitGroup(
                xyzuvw_dict=star_pars, burnin_steps=BURNIN_STEPS,
                plot_it=True, pool=pool, convergence_tol=C_TOL,
                plot_dir=gdir, save_dir=gdir, z=z[:, i],
                init_pos=all_init_pos[i],
                init_pars=all_init_pars,
            )
            logging.info("Finished fit")
            new_group = syn.Group(best_fit, sphere=True, internal=True,
                                  starcount=False)
            new_groups.append(new_group)
            np.save(gdir + "best_group_fit.npy", new_group)
            np.save(gdir+'final_chain.npy', chain)
            np.save(gdir+'final_lnprob.npy', lnprob)
            all_samples.append(chain)
            all_lnprob.append(lnprob)
            all_init_pos[i] = chain[:, -1, :]

        converged = checkConvergence(old_best_fits=old_groups,
                                     new_chains=all_samples,
                                     #perc=45, # COMMENT OUT THIS LINE
                                     #          # FOR LEGIT FITS!
                                     )
        logging.info("Convergence status: {}".format(converged))
        old_old_groups = old_groups
        old_groups = new_groups

        iter_count += 1

    logging.info("CONVERGENCE COMPLETE")

    np.save(rdir+"final_groups.npy", new_groups)
    np.save(rdir+"prev_groups.npy", old_old_groups) # old grps overwritten by new grps
    np.save(rdir+"memberships.npy", z)

    # PERFORM FINAL EXPLORATION OF PARAMETER SPACE
    final_dir = rdir+"final/"
    mkpath(final_dir)

    final_z = expectation(star_pars, new_groups, z)
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
            # init_pars=old_groups[i],
        )
        # run with extremely large convergence tolerance to ensure it only
        # runs once
        logging.info("Finished fit")
        final_best_fits[i] = best_fit
        final_med_errs[i] = calcErrors(chain)
        np.save(final_gdir + 'final_chain.npy', chain)
        np.save(final_gdir + 'final_lnprob.npy', lnprob)

        all_init_pos[i] = chain[:, -1, :]

    final_groups = [syn.Group(final_best_fit, sphere=True, internal=True,
                              starcount=False)
                    for final_best_fit in final_best_fits]
    np.save(final_dir+'final_groups.npy', final_groups)
    np.save(final_dir+'final_med_errs.npy', final_med_errs)

    logging.info("FINISHED CHARACTERISATION")
    #logging.info("Origin:\n{}".format(origins))
    logging.info("Best fits:\n{}".format(new_groups))
    logging.info("Memberships: \n{}".format(z))

    return final_best_fits, final_med_errs, z

