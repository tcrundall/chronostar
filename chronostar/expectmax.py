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
        upper_contained = old_best_fit < errors[:, 1]
        lower_contained = old_best_fit > errors[:, 2]

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
        weight = max(old_z[:,i].sum(), nstars/(2. * (ngroups+1)))
        group_pars = group.getInternalSphericalPars()
        lnols[:, i] =\
            weight*gf.lnlike(group_pars, star_pars, return_lnols=True)
        # calc_lnoverlaps(group_pars, star_pars, nstars)
    z = np.zeros((nstars, ngroups))
    for i in range(nstars):
        z[i] = calcMembershipProbs(lnols[i])
    if np.isnan(z).any():
        import pdb; pdb.set_trace()
    return z


def maximise(infile, ngroups, z=None, init_conditions=None,
             burnin_steps=500, sampling_steps=1000):
    """Given membership probabilities, maximise the parameters of each model

    Parameters
    ----------
    infile : str
        Name of the traceback file being fitted to

    ngroups : int
        Number of groups to be fitted to the traceback orbits

    z : [nstars, ngroups] array
        An array designating each star's probability of being a member to
        each group. It is populated by floats in the range (0.0, 1.0) such
        that each row sums to 1.0, each column sums to the expected size of
        each group, and the entire array sums to the number of stars.

    init_conditions : [ngroups, npars] array
        The initial conditions for the groups, encoded in the 'internal'
        manner (that is, 1/dX, 1/dY etc. and no star count)

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
        best_fit, chain, lnprob = tfgf.fit_group(
            infile, z=ix_snd(z, i), init_pars=ix_fst(init_conditions, i),
            burnin_steps=burnin_steps, plot_it=True
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

    mid_point = np.mean(xyzuvw, axis=0)[3:5]
    meanXYZ = np.array([0.,0.,0.])
    meanW = 0.
    dx = 50.
    dv = 5.
    age = 3.
    group_pars_base = list([0, 0, 0, None, None, 0, np.log(50),
                            np.log(5), 3])
    pts = getPointsOnCircle(npoints=ngroups, v_dist=110, offset=offset)

    for i in range(ngroups):
        meanUV = mid_point + pts[i]
        group_pars = np.hstack((meanXYZ, meanUV, meanW, dx, dv, age))
        group = syn.Group(group_pars, sphere=True, starcount=False)
        groups.append(group)

    return groups

def calcMnsCovs(new_groups, ngroups, origins=None):
    """
    Used for plotting... extracts means and covs form list of best fits

    Paramters
    ---------
        new_groups : [ngroups, npars] list
            best fits from the final run
        ngroups : int
            number of groups
        origins : [ngroups] synthesiser.Group object list {None}
            list of the Group objects corresponding to the intiialisation
            of the stars. (only applicable for a synthetic run)
    """
    all_origin_mn_then = [None] * ngroups
    all_origin_cov_then = [None] * ngroups
    all_origin_mn_now = [None] * ngroups
    all_origin_cov_now = [None] * ngroups
    all_fitted_mn_then = [None] * ngroups
    all_fitted_cov_then = [None] * ngroups
    all_fitted_mn_now = [None] * ngroups
    all_fitted_cov_now = [None] * ngroups

    for i in range(ngroups):
        if origins:
            all_origin_mn_then[i] = origins[i][:6]
            all_origin_cov_then[i] = utils.generate_cov(
                utils.internalise_pars(origins[i])
            )
            all_origin_mn_now[i] = tb.trace_forward(all_origin_mn_then[i],
                                                    origins[i][-2])
            all_origin_cov_now[i] = tf.transform_cov(
                all_origin_cov_then[i], tb.trace_forward,
                all_origin_mn_then[i],
                dim=6, args=(origins[i][-2],)
            )
        all_fitted_mn_then[i] = new_groups[i][:6]
        all_fitted_cov_then[i] = tfgf.generate_cov(new_groups[i])
        all_fitted_mn_now[i] = tb.trace_forward(all_fitted_mn_then[i],
                                                new_groups[i][-1])
        all_fitted_cov_now[i] = tf.transform_cov(
            all_fitted_cov_then[i], tb.trace_forward, all_fitted_mn_then[i],
            dim=6, args=(new_groups[i][-1],)
        )

    if origins:
        all_origin_mn_then  = np.array(all_origin_mn_then )
        all_origin_cov_then = np.array(all_origin_cov_then)
        all_origin_mn_now   = np.array(all_origin_mn_now  )
        all_origin_cov_now  = np.array(all_origin_cov_now )
    all_fitted_mn_then  = np.array(all_fitted_mn_then )
    all_fitted_cov_then = np.array(all_fitted_cov_then)
    all_fitted_mn_now   = np.array(all_fitted_mn_now  )
    all_fitted_cov_now  = np.array(all_fitted_cov_now )

    all_means = {
        'fitted_then' : all_fitted_mn_then,
        'fitted_now'  : all_fitted_mn_now ,
    }

    all_covs = {
        'fitted_then' : all_fitted_cov_then,
        'fitted_now'  : all_fitted_cov_now ,
    }

    if origins:
        all_means['origin_then'] = all_origin_mn_then
        all_means['origin_now']  = all_origin_mn_now
        all_covs['origin_then']  = all_origin_cov_then
        all_covs['origin_now']   = all_origin_cov_now

    np.save("means.npy", all_means)
    np.save("covs.npy", all_covs)

    return all_means, all_covs


def plotAll(star_pars, means, covs, ngroups, iter_count):
    plt.clf()
    xyzuvw = star_pars['xyzuvw'][:, 0]
    xyzuvw_cov = star_pars['xyzuvw_cov'][:, 0]
    plt.plot(xyzuvw[:, 0], xyzuvw[:, 1], 'b.')
    for mn, cov in zip(xyzuvw, xyzuvw_cov):
        ee.plot_cov_ellipse(cov[:2, :2], mn[:2], color='b',
                            alpha=0.3)
    for i in range(ngroups):
        ee.plot_cov_ellipse(covs['origin_then'][i][:2, :2],
                            means['origin_then'][i][:2], color='orange',
                            alpha=0.3, hatch='|', ls='--')
        #ee.plot_cov_ellipse(covs['origin_now'][i][:2, :2],
        #                    means['origin_now'][i][:2], color='xkcd:gold',
        #                    alpha=0.1, hatch='|', ls='--')
        ee.plot_cov_ellipse(covs['fitted_then'][i][:2, :2],
                            means['fitted_then'][i][:2],
                            color='xkcd:neon purple',
                            alpha=0.3, hatch='/', ls='-.')
        ee.plot_cov_ellipse(covs['fitted_now'][i][:2, :2],
                            means['fitted_now'][i][:2],
                            color='b',
                            alpha=0.1, hatch='.')
    min_means = np.min(np.array(means.values()).reshape(-1,6), axis=0)
    max_means = np.max(np.array(means.values()).reshape(-1,6), axis=0)

    xmin = min(min_means[0], np.min(xyzuvw[:,0]))
    xmax = max(max_means[0], np.max(xyzuvw[:,0]))
    ymin = min(min_means[1], np.min(xyzuvw[:,1]))
    ymax = max(max_means[1], np.max(xyzuvw[:,1]))

    buffer = 20
    plt.xlim(xmax+buffer, xmin-buffer)
    plt.ylim(ymin-buffer, ymax+buffer)

    plt.title("Iteration: {}".format(iter_count))
    plt.savefig("XY_plot.pdf", bbox_inches='tight', format='pdf')

    logging.info("Iteration {}: XY plot plotted".format(iter_count))

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
    BURNIN_STEPS = 10
    SAMPLING_STEPS = 50
    C_TOL = 50 # 0.5

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

    while not converged:
        # for iter_count in range(10):
        idir = rdir+"iter{}/".format(iter_count)
        logging.info("Iteration {}".format(iter_count))

        mkpath(idir)
        #os.chdir("iter{}".format(iter_count))

        # EXPECTATION
        z = expectation(star_pars, old_groups, z)
        np.save(idir+"membership.npy", z)

        # MAXIMISE
        #new_groups = np.zeros(old_groups.shape)
        new_groups = []

        all_samples = []
        all_lnprob = []

        for i in range(ngroups):
            logging.info("Fitting group {}".format(i))
            gdir = idir + "group{}/".format(i)
            mkpath(gdir)

            best_fit, chain, lnprob = gf.fitGroup(
                xyzuvw_dict=star_pars, burnin_steps=BURNIN_STEPS,
                plot_it=True, pool=pool, convergence_tol=C_TOL,
                plot_dir=gdir, save_dir=gdir, z=z[:, i],
                init_pos=all_init_pos[i],
                init_pars=old_groups[i].getInternalSphericalPars(),
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

        # ----- PLOTTING --------- #
        # means, covs = calcMnsCovs(new_groups, ngroups, origins=origins)
        #plot_all(star_pars, means, covs, ngroups, iter_count)
        # hp.plot_hexplot(star_pars, means, covs, iter_count)

        converged = checkConvergence(old_best_fits=old_groups,
                                     new_chains=all_samples,
                                     perc=45, # COMMENT OUT THIS LINE
                                               # FOR LEGIT FITS!
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

