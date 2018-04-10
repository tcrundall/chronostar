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

import tfgroupfitter as tfgf
from chronostar import utils
import chronostar.transform as tf
import chronostar.tracingback as tb
import chronostar.error_ellipse as ee
import chronostar.hexplotter as hp

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


def calc_errors(chain, perc=34):
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
                        zip(*np.percentile(flat_chain, [50-perc, 50, 50+perc], axis=0))))


def check_convergence(old_best_fits, new_chains,
                      tol=1.0):
    """Check if the last maximisation step yielded is consistent fit to new fit

    TODO: incorporate Z into this convergence checking. e.g.
    np.allclose(z_prev, z, rtol=1e-2)

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
    each_converged = []

    for old_best_fit, new_chain in zip(old_best_fits, new_chains):
        errors = calc_errors(new_chain, perc=25)
        upper_contained = old_best_fit < errors[:, 1]
        lower_contained = old_best_fit > errors[:, 2]

        each_converged.append(
            np.all(upper_contained) and np.all(lower_contained))

    return np.all(each_converged)


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
    lnols = None
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
        lnols[:, i] = tfgf.lnlike(group_pars, star_pars, return_lnols=True)
        # calc_lnoverlaps(group_pars, star_pars, nstars)
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


def get_points_on_circle(npoints, v_dist=20):
    """
    Little tool to found coordinates of equidistant points around a circle

    Used to initialise UV for the groups.
    :param npoints:
    :return:
    """
    us = np.zeros(npoints)
    vs = np.zeros(npoints)

    for i in range(npoints):
        us[i] = v_dist * np.cos(2 * np.pi * i / npoints)
        vs[i] = v_dist * np.sin(2 * np.pi * i / npoints)

    return np.vstack((us, vs)).T


def get_initial_group_pars(ngroups):
    """
    Generate the parameter list with which walkers will be initialised
    TODO: CENTRE THIS BY MEAN U AND V OF STARS

    :param ngroups:
    :return:
    """
    group_pars_base = list([0, 0, 0, None, None, 0, np.log(50), np.log(5), 3])
    pts = get_points_on_circle(npoints=ngroups, v_dist=110)

    all_init_group_pars = np.array(
        ngroups * group_pars_base
    ).reshape(-1, len(group_pars_base))

    for init_group_pars, pt in zip(all_init_group_pars, pts):
        init_group_pars[3:5] = pt
    return np.array(all_init_group_pars, dtype=np.float64)

def calc_mns_covs(new_gps, ngroups, origins=None):
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
                all_origin_cov_then[i], tb.trace_forward, all_origin_mn_then[i],
                dim=6, args=(origins[i][-2],)
            )
        all_fitted_mn_then[i] = new_gps[i][:6]
        all_fitted_cov_then[i] = tfgf.generate_cov(new_gps[i])
        all_fitted_mn_now[i] = tb.trace_forward(all_fitted_mn_then[i],
                                                new_gps[i][-1])
        all_fitted_cov_now[i] = tf.transform_cov(
            all_fitted_cov_then[i], tb.trace_forward, all_fitted_mn_then[i],
            dim=6, args=(new_gps[i][-1],)
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


def plot_all(star_pars, means, covs, ngroups, iter_count):
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

def fit_multi_groups(star_pars, ngroups, res_dir='', init_z=None, origins=None):
    """
    Entry point: Fit multiple Gaussians to data set

    :param star_pars:
    :param ngroups:
    :return:
    """
    if res_dir:
        os.chdir(res_dir)

    # INITIALISE GROUPS
    old_gps = get_initial_group_pars(ngroups)

    all_init_pos = ngroups * [None]
    iter_count = 0
    converged = False
    while not converged:
        # for iter_count in range(10):
        logging.info("Iteration {}".format(iter_count))
        mkpath("iter{}".format(iter_count))
        os.chdir("iter{}".format(iter_count))

        # EXPECTATION
        z = expectation(star_pars, old_gps)
        np.save("membership.npy", z)

        # MAXIMISE
        new_gps = np.zeros(old_gps.shape)

        all_samples = []
        all_lnprob = []

        for i in range(ngroups):
            logging.info("Fitting group {}".format(i))
            try:
                mkpath("group{}".format(i))
                os.chdir("group{}".format(i))
            except:
                pathname = random.shuffle("apskjfa")
                mkpath(pathname)
                os.chdir(pathname)
            best_fit, samples, lnprob = tfgf.fit_group(
                "../../perf_tb_file.pkl", z=z[:, i], burnin_steps=50, #burnin was 500
                plot_it=True,
                init_pars=old_gps[i], convergence_tol=50., tight=True, #tol was 5.
                init_pos=all_init_pos[i])
            logging.info("Finished fit")
            new_gps[i] = best_fit
            np.save('final_chain.npy', samples)
            np.save('final_lnprob.npy', lnprob)
            all_samples.append(samples)
            all_lnprob.append(lnprob)
            all_init_pos[i] = samples[:, -1, :]
            os.chdir("..")

        # ----- PLOTTING --------- #
        means, covs = calc_mns_covs(new_gps, ngroups, origins=origins)
        #plot_all(star_pars, means, covs, ngroups, iter_count)
        hp.plot_hexplot(star_pars, means, covs, iter_count)

        converged = check_convergence(old_best_fits=old_gps,
                                      new_chains=all_samples)
        logging.info("Convergence status: {}".format(converged))
        old_old_gps = old_gps
        old_gps = new_gps

        os.chdir("..")
        iter_count += 1
    logging.info("CONVERGENCE COMPLETE")

    np.save("final_gps.npy", new_gps)
    np.save("prev_gps.npy", old_old_gps) # old groups is overwritten by new grps
    np.save("memberships.npy", z)

    # PERFORM FINAL EXPLORATION OF PARAMETER SPACE
    mkpath("final")
    os.chidr("final")
    final_z = expectation(star_pars, new_gps)
    final_gps = [None] * ngroups
    final_med_errs = [None] * ngroups

    for i in range(ngroups):
        logging.info("Characterising group {}".format(i))
        try:
            mkpath("group{}".format(i))
            os.chdir("group{}".format(i))
        except:
            pathname = random.shuffle("apskjfa")
            mkpath(pathname)
            os.chdir(pathname)
        best_fit, samples, lnprob = tfgf.fit_group(
            "../../perf_tb_file.pkl", z=z[:, i], burnin_steps=2000,
            plot_it=True,
            init_pars=old_gps[i], convergence_tol=20., tight=True,  # tol was 10
            init_pos=all_init_pos[i])
        logging.info("Finished fit")
        final_gps[i] = best_fit
        final_med_errs[i] = calc_errors(samples)
        np.save('final_chain.npy', samples)
        np.save('final_lnprob.npy', lnprob)

        all_init_pos[i] = samples[:, -1, :]
        os.chdir("..")
    np.save('final_med_errs.npy', final_med_errs)
    logging.info("FINISHED CHARACTERISATION")
    logging.info("Origin:\n{}".format(origins))
    logging.info("Best fits:\n{}".format(new_gps))
    logging.info("Memberships: \n{}".format(z))


if __name__ == "__main__":
    # TODO: ammend calc_lnols such that group sizes impact membership probs

    if len(sys.argv) > 1:
        sys.path.insert(0, sys.argv[1])

    logging.basicConfig(
        level=logging.DEBUG, filemode='w',
        filename='em.log',
    )

    origins = np.array([
       #  X    Y    Z    U    V    W   dX  dY    dZ  dVCxyCxzCyz age nstars
       [25., 0., 11., -5., 0., -2., 10., 10., 10., 5., 0., 0., 0., 3., 50.],
       [-21., -60., 4., 3., 10., -1., 7., 7., 7., 3., 0., 0., 0., 7., 30.],
#       [-10., 20., 0., 1., -4., 15., 10., 10., 10., 2., 0., 0., 0., 10., 40.],
#       [-80., 80., -80., 5., -5., 5., 20., 20., 20., 5., 0., 0., 0., 13., 80.],

    ])
    ERROR = 1.0

    ngroups = origins.shape[0]
    TB_FILE = "perf_tb_file.pkl"
    astr_file = "perf_astr_data.pkl"

    logging.info("Origin:\n{}".format(origins))
    np.save("origins.npy", origins)
    perf_xyzuvws, _ = syn.generate_current_pos(ngroups, origins)

    np.save("perf_xyzuvw.npy", perf_xyzuvws)
    sky_coord_now = syn.measure_stars(perf_xyzuvws)

    synth_table = syn.generate_table_with_error(
        sky_coord_now, ERROR
    )

    pickle.dump(synth_table, open(astr_file, 'w'))
    tb.traceback(synth_table, np.array([0, 1]), savefile=TB_FILE)
    star_pars = tfgf.read_stars(TB_FILE)

    fit_multi_groups(star_pars, ngroups)

