from __future__ import division, print_function

import numpy as np

from astropy.table import Table
import emcee
import logging

from chronostar.component import SphereComponent as Component
from chronostar.likelihood import lnprob_func
from chronostar import tabletool

try:
    import matplotlib.pyplot as plt
    plt_avail = True
except ImportError:
    plt_avail = False


def calcMedAndSpan(chain, perc=34, intern_to_extern=False, sphere=True):
    """
    Given a set of aligned samples, calculate the 50th, (50-perc)th and
     (50+perc)th percentiles.

    Parameters
    ----------
    chain : [nwalkers, nsteps, npars] array -or- [nwalkers*nsteps, npars] array
        The chain of samples (in internal encoding)
    perc: integer {34}
        The percentage from the midpoint you wish to set as the error.
        The default is to take the 16th and 84th percentile.
    intern_to_extern : boolean {False}
        Set to true if chain has dx and dv provided in log form and output
        is desired to have dx and dv in linear form
    sphere: Boolean {True}
        Currently hardcoded to take the exponent of the logged
        standard deviations. If sphere is true the log stds are at
        indices 6, 7. If sphere is false, the log stds are at
        indices 6:10.

    Returns
    -------
    result : [npars,3] float array
        For each parameter, there is the 50th, (50+perc)th and (50-perc)th
        percentiles
    """
    npars = chain.shape[-1]  # will now also work on flatchain as input
    flat_chain = np.reshape(chain, (-1, npars))

    if intern_to_extern:
        # take the exponent of the log(dx) and log(dv) values
        flat_chain = np.copy(flat_chain)
        if sphere:
            flat_chain[:, 6:8] = np.exp(flat_chain[:, 6:8])
        else:
            flat_chain[:, 6:10] = np.exp(flat_chain[:, 6:10])

    return np.array(map(lambda v: (v[1], v[2], v[0]),
                        zip(*np.percentile(flat_chain,
                                           [50-perc, 50, 50+perc],
                                           axis=0))))


def approxCurrentDayDistribution(data, membership_probs):
    means = tabletool.buildDataFromTable(data, cartesian=True,
                                         only_means=True)
    mean_of_means = np.average(means, axis=0, weights=membership_probs)
    cov_of_means = np.cov(means.T, ddof=0., aweights=membership_probs)
    return mean_of_means, cov_of_means


def noStuckWalkers(lnprob):
    """
    Examines lnprob to see if any walkers have flatlined far from pack

    TODO: rewrite this using 'percentile' to set some range
          i.e. no walker should be more than 3*D from mean where
          D is np.perc(final_pos, 50) - np.perc(final_pos, 16)
    """
    final_pos = lnprob[:,-1]
    std_proxy = np.percentile(final_pos, 50) -\
                np.percentile(final_pos, 16)

    worst_walker = np.min(final_pos)
    res = worst_walker > np.percentile(final_pos, 50) - 3*std_proxy
    logging.info("No stuck walkers? {}".format(res))
    return res


def burninConvergence(lnprob, tol=0.25, slice_size=100, cutoff=0):
    """Checks early lnprob vals with final lnprob vals for convergence

    Parameters
    ----------
    lnprob : [nwalkers, nsteps] array

    tol : float
        The number of standard deviations the final mean lnprob should be
        within of the initial mean lnprob

    slice_size : int
        Number of steps at each end to use for mean lnprob calcultions

    cuttoff : int
        Step number at which to start the analysis. i.e., a value of 0 would
        mean to include the whole chain, whereas a value of 500 would be to
        only confirm no deviation in the steps [500:END]
    """
    # take a chunk the smaller of 100 or half the chain
    if lnprob.shape[1] <= 2*slice_size:
        logging.info("Burnin length {} too small for reliable convergence"
                     "checking".\
                     format(lnprob.shape[1]))
        slice_size = int(round(0.5*lnprob.shape[1]))

    start_lnprob_mn = np.mean(lnprob[:,:slice_size])
    #start_lnprob_std = np.std(lnprob[:,:slice_size])

    end_lnprob_mn = np.mean(lnprob[:, -slice_size:])
    end_lnprob_std = np.std(lnprob[:, -slice_size:])

    return (np.isclose(start_lnprob_mn, end_lnprob_mn,
                       atol=tol*end_lnprob_std)
            and noStuckWalkers(lnprob))


def getInitEmceePos(data, memb_probs=None, nwalkers=None,
                    init_pars=None, model_form='sphere'):
    if init_pars is None:
        rough_mean_now, rough_cov_now = \
            approxCurrentDayDistribution(data=data, membership_probs=memb_probs)
        # Exploit the component logic to generate closest set of pars
        dummy_comp = Component(mean=rough_mean_now,
                               covmatrix=rough_cov_now,
                               form=model_form)
        init_pars = Component.internalisePars(dummy_comp.getPars())

    init_std = Component.getSensibleInitSpread(form=model_form)

    # Generate initial positions of all walkers by adding some random
    # offset to `init_pars`
    if nwalkers is None:
        npars = len(init_pars)
        nwalkers = 2 * npars
    init_pos = emcee.utils.sample_ball(init_pars, init_std,
                                       size=nwalkers)
    # force ages to be positive
    init_pos[:, -1] = abs(init_pos[:, -1])
    return init_pos


def fitGroup(data=None, memb_probs=None, burnin_steps=1000, model_form='sphere',
             plot_it=False, pool=None, convergence_tol=0.25, init_pos=None,
             plot_dir='', save_dir='', init_pars=None, sampling_steps=None,
             max_iter=None):
    """Fits a single gaussian to a weighted set of traceback orbits.

    Stores the final sampling chain and lnprob in `save_dir`, but also
    returns the best fit (walker step corresponding to maximum lnprob),
    sampling chain and lnprob.

    Parameters
    ----------
    xyzuvw_dict, xyzuvw_file : (dict or string)
        Can either pass in the dictionary directly, or a '.fits' filename
        from which to load the dictionary:
            xyzuvw : (nstars,ntimes,6) np array
                mean XYZ in pc and UVW in km/s for stars
            xyzuvw_cov : (nstars,ntimes,6,6) np array
                covariance of xyzuvw
            table: (nstars) high astropy table including columns as
                        documented in the measurer class.
    z : ([nstars] array {None})
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    burnin_steps : int {1000}
        Number of steps per each burnin iteration

    plot_it : bool {False}
        Whether to generate plots of the lnprob in 'plot_dir'

    pool : MPIPool object {None}
        pool of threads to execute walker steps concurrently

    convergence_tol : float {0.25}
        How many standard devaitions an lnprob chain is allowed to vary
        from its mean over the course of a burnin stage to be considered
        "converged". Default value allows the median of the final 20 steps
        to differ by 0.25 of its standard devations from the median of the
        first 20 steps.

    init_pos : [ngroups, npars] array
        The precise locations at which to initiate the walkers. Generally
        the saved locations from a previous, yet similar run.

    plot_dir : str {''}
        The directory in which to save plots

    save_dir : str {''}
        The directory in which to store results and/or byproducts of fit

    init_pars : [npars] array
        the position in parameter space about which walkers should be
        initialised. The standard deviation about each parameter is
        hardcoded as INIT_SDEV

    sampling_steps : int {None}
        If this is set, after convergence, a sampling stage will be
        entered. Only do this if a very fine map of the parameter
        distributions is required, since the burnin stage already
        characterises a converged solution for "burnin_steps".

    traceback : bool {False}
        Set this as true to run with the (retired) traceback
        implementation.

    max_iter : int {None}
        The maximum iterations permitted to run. (Useful for expectation
        maximisation implementation triggering an abandonment of rubbish
        components)

    init_at_mean : bool {False}
        If set, the walkers will be initialised around the mean of the
        provided data set. (Not recommended for multi-component fits)

    Returns
    -------
    best_sample
        The parameters of the group model which yielded the highest
        posterior probability

    chain
        [nwalkers, nsteps, npars] array of all samples

    probability
        [nwalkers, nsteps] array of probabilities for each sample
    """
    if isinstance(data, str):
        data = Table.read(data)
    if memb_probs is None:
        memb_probs = np.ones(len(data))

    # initialise z if needed as array of 1s of length nstars
    if memb_probs is None:
        memb_probs = np.ones(len(data))

    # Initialise the fit
    if init_pos is None:
        init_pos = getInitEmceePos(data=data, memb_probs=memb_probs,
                                   init_pars=init_pars,
                                   model_form=model_form)
    nwalkers, npars = init_pos.shape

    # Whole emcee shebang
    sampler = emcee.EnsembleSampler(
        nwalkers, npars, lnprob_func, args=[data, memb_probs], pool=pool,
    )

    # Perform burnin
    state = None
    converged = False
    cnt = 0
    logging.info("Beginning burnin loop")
    burnin_lnprob_res = np.zeros((nwalkers,0))

    # burn in until converged or the optional max_iter is reached
    while (not converged) and cnt != max_iter:
        logging.info("Burning in cnt: {}".format(cnt))
        sampler.reset()
        init_pos, lnprob, state = sampler.run_mcmc(init_pos, burnin_steps, state)
        converged = burninConvergence(sampler.lnprobability,
                                      tol=convergence_tol)
        logging.info("Burnin status: {}".format(converged))

        if plot_it and plt_avail:
            plt.clf()
            plt.plot(sampler.lnprobability.T)
            plt.savefig(plot_dir+"burnin_lnprobT{:02}.png".format(cnt))

        # If about to burnin again, help out the struggling walkers
        if not converged:
            best_ix = np.argmax(lnprob)
            poor_ixs = np.where(lnprob < np.percentile(lnprob, 33))
            for ix in poor_ixs:
                init_pos[ix] = init_pos[best_ix]

        burnin_lnprob_res = np.hstack((
            burnin_lnprob_res, sampler.lnprobability
        ))
        cnt += 1

    logging.info("Burnt in, with convergence: {}".format(converged))
    if plot_it and plt_avail:
        plt.clf()
        plt.plot(burnin_lnprob_res.T)
        plt.savefig(plot_dir+"burnin_lnprobT.png")

    if not sampling_steps:
        logging.info("Taking final burnin segment as sampling stage"\
                     .format(converged))
    else:
        logging.info("Entering sampling stage for {} steps".format(
            sampling_steps
        ))
        sampler.reset()
        init_pos, lnprob, state = sampler.run_mcmc(init_pos, sampling_steps,
                                                   state)
        logging.info("Sampling done")

    # save the chain for later inspection
    np.save(save_dir+"final_chain.npy", sampler.chain)
    np.save(save_dir+"final_lnprob.npy", sampler.lnprobability)

#    print("Sampled")
    if plot_it and plt_avail:
        logging.info("Plotting final lnprob")
        plt.clf()
        plt.plot(sampler.lnprobability.T)
        plt.savefig(plot_dir+"lnprobT.png")
        logging.info("Plotting done")

    # sampler.lnprobability has shape (NWALKERS, SAMPLE_STEPS)
    # yet np.argmax takes index of flattened array
    final_best_ix = np.argmax(sampler.lnprobability)
    best_sample = sampler.flatchain[final_best_ix]

    # displaying median and range of each paramter
    med_and_span = calcMedAndSpan(sampler.chain)

    logging.info("Results:\n{}".format(med_and_span))

    return best_sample, sampler.chain, sampler.lnprobability


