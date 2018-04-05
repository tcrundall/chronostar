"""
a module for implementing the expectation-maximisation algorithm
in order to fit a multi-gaussian mixture model of moving groups' origins
to a data set of stars tracedback through XYZUVW

todo:
    - implement average error cacluation in lnprobfunc
"""
from __future__ import print_function, division

import sys
import numpy as np

try:
    import matplotlib as mpl

    # prevents displaying plots from generation from tasks in background
    mpl.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not imported")
    pass

import tfgroupfitter as tfgf
import pdb  # for debugging

# for permuting samples when realigning
try:
    import astropy.io.fits as pyfits
except:
    import pyfits

try:
    import _overlap as overlap  # &TC
except:
    print(
        "overlap not imported, SWIG not possible. Need to make in directory...")

try:  # don't know why we use xrange to initialise walkers
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
        return array[:, ix]


def calc_errors(chain):
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
                        zip(*np.percentile(flat_chain, [16, 50, 84], axis=0))))


def check_convergence(old_best_fits, new_chains,
                      tol=1.0):
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
    each_converged = []

    for old_best_fit, new_chain in zip(old_best_fits, new_chains):
        errors = calc_errors(new_chain)
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


def run_fit(infile, nburnin, nsteps, ngroups=1):
    """
    Entry point for module. Given the traceback of stars, fit a set number
    of groups to the data.
    """


#    star_params = gf.read_stars(infile)
#    samples, lnprob = gf.fit_one_group(star_params, nburnin, nsteps)


def get_initial_group_pars(ngroups):
    """
    Generate the parameter list with which walkers will be initialised

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

def calc_mns_covs(origins, new_gps, ngroups):
    all_origin_mn_then = [None] * ngroups
    all_origin_cov_then = [None] * ngroups
    all_origin_mn_now = [None] * ngroups
    all_origin_cov_now = [None] * ngroups
    all_fitted_mn_then = [None] * ngroups
    all_fitted_cov_then = [None] * ngroups
    all_fitted_mn_now = [None] * ngroups
    all_fitted_cov_now = [None] * ngroups

    for i in range(ngroups):
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
    all_origin_mn_then  = np.array(all_origin_mn_then )
    all_origin_cov_then = np.array(all_origin_cov_then)
    all_origin_mn_now   = np.array(all_origin_mn_now  )
    all_origin_cov_now  = np.array(all_origin_cov_now )
    all_fitted_mn_then  = np.array(all_fitted_mn_then )
    all_fitted_cov_then = np.array(all_fitted_cov_then)
    all_fitted_mn_now   = np.array(all_fitted_mn_now  )
    all_fitted_cov_now  = np.array(all_fitted_cov_now )

    all_means = {
        'origin_then' : all_origin_mn_then,
        'origin_now'  : all_origin_mn_now ,
        'fitted_then' : all_fitted_mn_then,
        'fitted_now'  : all_fitted_mn_now ,
    }

    all_covs = {
        'origin_then' : all_origin_cov_then,
        'origin_now'  : all_origin_cov_now ,
        'fitted_then' : all_fitted_cov_then,
        'fitted_now'  : all_fitted_cov_now ,
    }

    np.save("means.npy",
            [all_origin_mn_then, all_origin_mn_now, all_fitted_mn_then,
             all_fitted_mn_now])
    np.save("covs.npy",
            [all_origin_cov_then, all_origin_cov_now, all_fitted_cov_then,
             all_fitted_cov_now])

    return all_means, all_covs


def plot_all(star_pars, means, covs):
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
                            alpha=0.1, hatch='|', ls='--')
        #ee.plot_cov_ellipse(covs['origin_now'][i][:2, :2],
        #                    means['origin_now'][i][:2], color='xkcd:gold',
        #                    alpha=0.1, hatch='|', ls='--')
        ee.plot_cov_ellipse(covs['fitted_then'][i][:2, :2],
                            means['fitted_then'][i][:2],
                            color='xkcd:neon purple',
                            alpha=0.2, hatch='/', ls='-.')
        ee.plot_cov_ellipse(covs['fitted_now'][i][:2, :2],
                            means['fitted_now'][i][:2],
                            color='b',
                            alpha=0.03, hatch='.')

    xmin = min(np.min(means['origin_then'][:,0]),
                np.min(means['origin_now'][:,0]),
               np.min(means['fitted_then'][:,0]),
                np.min(means['fitted_now'][:,0]),
               )

    xmax = max(np.max(means['origin_then'][:,0]),
               np.max(means['origin_now'][:,0]),
               np.max(means['fitted_then'][:,0]),
               np.max(means['fitted_now'][:,0]),
               )

    ymin = min(np.min(means['origin_then'][:,1]),
               np.min(means['origin_now'][:,1]),
               np.min(means['fitted_then'][:,1]),
               np.min(means['fitted_now'][:,1]),
               )

    ymax = max(np.max(means['origin_then'][:,1]),
               np.max(means['origin_now'][:,1]),
               np.max(means['fitted_then'][:,1]),
               np.max(means['fitted_now'][:,1]),
               )

    buffer = 30
    plt.xlim(xmax+buffer, xmin-buffer)
    plt.ylim(ymin-buffer, ymax+buffer)

    plt.title("Iteration: {}".format(iter_count))
    plt.savefig("XY_plot.png")
    logging.info("Iteration {}: XY plot plotted".format(iter_count))

if __name__ == "__main__":
    # TODO: ammend calc_lnols such that group sizes impact membership probs
    from distutils.dir_util import mkpath
    import os
    import logging
    import random
    import chronostar.synthesiser as syn
    import pickle
    import chronostar.traceback as tb
    from chronostar import utils
    import chronostar.transform as tf
    import chronostar.error_ellipse as ee



    logging.basicConfig(
        level=logging.DEBUG, filemode='w',
        filename='em.log',
    )

    # initial conditions
    origins = np.array([
        [20., 0., 0., 10., 10., 0., 10., 10., 10., 5., 0.,
         0., 0., 3., 50.],
        [-10., 10., 0., -10., -10., 0., 10., 10., 10., 5., 0.,
         0., 0., 5., 30.]
        [10., -50., 0., -30., 50., 0., 5., 5., 5., 5., 0.,
         0., 0., 10., 60.]
    ])
    ERROR = 0.3

    ngroups = origins.shape[0]
    TB_FILE = "perf_tb_file.pkl"
    astr_file = "perf_astr_data.pkl"

    logging.info("Origin:\n{}".format(origins))

    np.save("origins.npy", origins)
    perf_xyzuvws, _ = syn.generate_current_pos(2, origins)

    perf_xyzuvws, _ = syn.generate_current_pos(2, origins)
    np.save("perf_xyzuvw.npy", perf_xyzuvws)
    sky_coord_now = syn.measure_stars(perf_xyzuvws)

    synth_table = syn.generate_table_with_error(
        sky_coord_now, ERROR
    )

    pickle.dump(synth_table, open(astr_file, 'w'))

    tb.traceback(synth_table, np.array([0, 1]), savefile=TB_FILE)

    star_pars = tfgf.read_stars(TB_FILE)

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
                "../../perf_tb_file.pkl", z=z[:, i], burnin_steps=300,
                plot_it=True,
                init_pars=old_gps[i], convergence_tol=10., tight=True,
                init_pos=all_init_pos[i])
            logging.info("Finished fit")
            new_gps[i] = best_fit
            all_samples.append(samples)
            all_lnprob.append(lnprob)
            all_init_pos[i] = samples[:, -1, :]
            os.chdir("..")

        # plot_all(new_gps, star_pars, origins, ngroups, iter_count)

        # ----- PLOTTING --------- #
        means, covs = calc_mns_covs(origins, new_gps, ngroups)
        plot_all(star_pars, means, covs)

        converged = check_convergence(old_best_fits=old_gps,
                                      new_chains=all_samples)
        logging.info("Convergence status: {}".format(converged))
        old_old_gps = old_gps
        old_gps = new_gps

        os.chdir("..")
        iter_count += 1
    logging.info("COMPLETE")
    logging.info("Origin:\n{}".format(origins))
    logging.info("Best fits:\n{}".format(new_gps))
    logging.info("Memberships: \n{}".format(z))
