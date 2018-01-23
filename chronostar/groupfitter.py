from __future__ import division, print_function

import numpy as np

from chronostar._overlap import get_lnoverlaps
import corner
import emcee
import matplotlib.pyplot as plt
import pickle
import utils

try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits

def weighted_mean(interp_mns, z):
    """Calculated the approximate X,Y,Z,U,V,W mean values of weighted stars

    Parameters
    ----------
    interp_mns : [nstars, 6] array
        interpolated means of the stars
    z : [nstars] array
        membership probabilities to group currently being fitted

    Returns
    -------
    - : [6] float array
        the mean values of the X,Y,Z,U,V,W
    """
    exp_star_cnt = np.sum(z)
    six_tiled_z = np.tile(z, (6,1))
    weighted_mns = np.multiply(six_tiled_z.T, interp_mns)
    return np.sum(weighted_mns, axis=0) / exp_star_cnt


def weighted_std(interp_mns, weighted_mns, z):
    """Calculated the approximate X,Y,Z,U,V,W stds of weighted stars

    dU, dV and dW are combined into a single average velocity dispersion

    Parameters
    ----------
    interp_mns : [nstars, 6] array
        interpolated means of the stars
    weighted_mns : [6] array
        output of weighted_mean
    z : [nstars] array
        membership probabilities to group currently being fitted

    Returns
    -------
    - : [4] float array
        the std values of X,Y,Z,V
    """
    exp_star_cnt = np.sum(z)
    six_tiled_z = np.tile(z, (6, 1))
    diffs = interp_mns - weighted_mns
    weighted_devs = np.multiply(six_tiled_z.T, np.square(diffs))
    weighted_stds = np.sqrt( np.sum(weighted_devs, axis=0) / exp_star_cnt)
    pos_stds = weighted_stds[:3]
    vel_std = np.power(np.prod(weighted_stds[3:]), 1.0/3.0)
    return np.append(pos_stds, vel_std)


def read_stars(tb_file):
    """Read stars from traceback file into a dictionary.

    The input is an error ellipse in 6D (X,Y,Z,U,V,W) of a list of stars at
    a bucn of times in the past.

    Parameters
    ----------
    tb_file: string
        input file with values to be wrapped as dictionary

    Returns
    -------
    star_dictionary : dict
        stars: (nstars) high astropy table including columns as
                    documented in the Traceback class.
        times: (ntimes) numpy array, containing times that have
                    been traced back, in Myr
        xyzuvw: (nstars,ntimes,6) numpy array, XYZ in pc and UVW in km/s
        xyzuvw_cov: (nstars,ntimes,6,6) numpy array, covariance of xyzuvw
    """
    if len(tb_file) == 0:
        print("Input a filename...")
        raise UserWarning

    # Stars is an astropy.Table of stars
    if tb_file[-3:] == 'pkl':
        with open(tb_file, 'r') as fp:
            (stars, times, xyzuvw, xyzuvw_cov) = pickle.load(fp)
    elif (tb_file[-3:] == 'fit') or (tb_file[-4:] == 'fits'):
        stars = pyfits.getdata(tb_file, 1)
        times = pyfits.getdata(tb_file, 2)
        xyzuvw = pyfits.getdata(tb_file, 3)
        xyzuvw_cov = pyfits.getdata(tb_file, 4)
    else:
        print("Unknown File Type!")
        raise UserWarning

    return dict(stars=stars, times=times, xyzuvw=xyzuvw, xyzuvw_cov=xyzuvw_cov)


def interp_cov(target_time, star_pars):
    """Calculates the xyzuvw vector and covariance matrix by interpolation
    
    Parameters
    ---------
    target_time
        The desired time to be fitted to
    star_pars
        Dictionary with
        xyzuvw
            [nstars, nts, 6] array with the phase values for each star
            at each traceback time
        xyzuvw_cov
            [nstars, nts, 6, 6] array with the covariance matrix of the
            phase values for each star at each traceback time
        times
            [nts] array with a linearly spaced times spanning 0 to some
            maximum time

    Returns
    -------
    interp_covs
        [nstars, 6, 6] array with covariance matrix of the phase values
        for each star at interpolated time
    interp_mns
        [nstars, 6] array with phase values for each star at interpolated
        time
    """
    times = star_pars['times']
    ix = np.interp(target_time, times, np.arange(len(times)))
    # check if interpolation is necessary
    if np.isclose(target_time, times, atol=1e-5).any():
        ix0 = int(round(ix))
        interp_mns = star_pars['xyzuvw'][:, ix0]
        interp_covs = star_pars['xyzuvw_cov'][:, ix0]
        return interp_covs, interp_mns

    ix0 = np.int(ix)
    frac = ix - ix0
    interp_covs = star_pars['xyzuvw_cov'][:, ix0] * (1 - frac) + \
                  star_pars['xyzuvw_cov'][:, ix0 + 1] * frac

    interp_mns = star_pars['xyzuvw'][:, ix0] * (1 - frac) + \
                 star_pars['xyzuvw'][:, ix0 + 1] * frac

    return interp_covs, interp_mns


def eig_prior(char_min, eig_val):
    """Computes the prior on the eigen values of the model Gaussain distr.

    Parameters
    ----------
    char_min
        The characteristic minimum of the position or velocity dispersions
    eig_val
        Eigen values of the covariance matrix representing of the 
        Gaussian distribution describing the group
    
    Returns
    -------
    eig_prior
        A prior on the provided model eigen value
    """
    # eig_val = 1 / inv_eig_val
    prior = eig_val / (char_min ** (2) + eig_val ** 2)
    return prior


def lnprior(pars, z, star_pars):
    """Computes the prior of the group models constraining parameter space

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
    z
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.
    star_pars
        traceback data being fitted to

    Returns
    -------
    lnprior
        The logarithm of the prior on the model parameters
    """
    # fetch maximum allowed age
    max_age = star_pars['times'][-1]

    means = pars[0:6]
    inv_stds = pars[6:10]
    corrs = pars[10:13]
    age = pars[13]
    if np.min(means) < -1000 or np.max(means) > 1000:
        return -np.inf
    if np.min(inv_stds) <= 0.0 or np.max(inv_stds) > 10.0:
        return -np.inf
    if np.min(corrs) < -1.0 or np.max(corrs) > 1.0:
        return -np.inf
    if age < 0.0 or age > max_age:
        return -np.inf
    return 0.0


def lnlike(pars, z, star_pars):
    """Computes the log-likelihood for a fit to a group.

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
    z
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.
    star_pars
        traceback data being fitted to

    Returns
    -------
    lnlike
        the logarithm of the likelihood of the fit
    """

    # convert pars into (in?)covariance matrix
    group_cov = utils.generate_cov(pars)

    # check if covariance matrix is singular
    if np.min(np.linalg.eigvalsh(group_cov)) < 0:
        return -np.inf

    group_mn = pars[0:6]

    # interpolate star data to modelled age
    age = pars[13]
    interp_covs, interp_mns = interp_cov(age, star_pars)
    nstars = interp_mns.shape[0]

    # PROGRESS HALTED! REWRITE C CODE IN LOGARITHMS BEFORE CONTINUING
    lnols = get_lnoverlaps(
        group_cov, group_mn, interp_covs, interp_mns, nstars
    )

    return np.sum(z * lnols)


def lnprobfunc(pars, z, star_pars):
    """Computes the log-probability for a fit to a group.

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
        0,1,2,3,4,5,   6,   7,   8,   9, 10, 11, 12, 13
        X,Y,Z,U,V,W,1/dX,1/dY,1/dZ,1/dV,Cxy,Cxz,Cyz,age
    z
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.
    star_pars
        traceback data being fitted to

    Returns
    -------
    logprob
        the logarithm of the posterior probability of the fit
    """
    #global N_FAILS
    #global N_SUCCS

    lp = lnprior(pars, z, star_pars)
    if not np.isfinite(lp):
        #N_FAILS += 1
        return -np.inf
    #N_SUCCS += 1
    return lp + lnlike(pars, z, star_pars)

def get_initial_pars(star_pars, initial_age, z):
    """Initialise the walkers around the mean XYZUVW of the stars

    Having this step cuts number of steps required for convergence by
    a few hundred for small data sets (~40 stars) and a few thousand
    for larger data sets (~8,000 stars)

    TODO: maybe have some means to incorporate membership probabilities

    :param star_pars:
    :param initial_age:
    :return:
    """
    #            X,Y,Z,U,V,W,1/dX,1/dY,1/dZ,1/dV,Cxy,Cxz,Cyz,age
    # init_pars = [0,0,0,0,0,0, 0.1, 0.1, 0.1, 0.2,0.0,0.0,0.0,2.0]
    init_pars = np.zeros(14)
    _, interp_mns = interp_cov(initial_age, star_pars)

    wmns = weighted_mean(interp_mns, z)
    init_pars[0:6] = wmns
    init_pars[6:10] = 1.0 / weighted_std(interp_mns, wmns, z)
    return init_pars

def fit_group(tb_file, z=None, burnin_steps=500, sampling_steps=1000,
              init_pars=None, plot_it=False, fixed_age=None):
    """Fits a single gaussian to a weighted set of traceback orbits.

    Parameters
    ----------
    tb_file : string
        a '.pkl' or '.fits' file containing traceback orbits as a dictionary:
            stars: (nstars) high astropy table including columns as
                        documented in the Traceback class.
            times: (ntimes) numpy array, containing times that have
                        been traced back, in Myr
            xyzuvw (nstars,ntimes,6) numpy array, XYZ in pc and UVW in km/s
            xyzuvw_cov (nstars,ntimes,6,6) numpy array, covariance of xyzuvw
    z
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    fixed_age : float
        can optionally fix the age of group to some value. Walkers will
        remain static on that age

    init_pars : [14]
        optionally can initialise the walkers around this point
    
    Returns
    -------
    best_sample
        The parameters of the group model which yielded the highest posterior
        probability

    chain
        [nwalkers, nsteps, npars] array of all samples

    probability
        [nwalkers, nsteps] array of probabilities for each sample
    """
    #global N_FAILS
    #global N_SUCCS

    #            X,Y,Z,U,V,W,1/dX,1/dY,1/dZ,1/dV,Cxy,Cxz,Cyz,age
    INIT_SDEV = [1, 1, 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.02, 0.1, 0.1, 0.1, 0.5]
    star_pars = read_stars(tb_file)
    if fixed_age is None:
        initial_age = 0.0
    else:
        initial_age = fixed_age

    # initialise z if needed as array of 1s of length nstars
    if z is None:
        z = np.ones(star_pars['xyzuvw'].shape[0])

    # Initialise the fit
    if init_pars is None:
        init_pars = get_initial_pars(star_pars, initial_age, z)

    NPAR = len(init_pars)
    NWALKERS = 2 * NPAR

    # Since emcee linearly interpolates between walkers to determine next step
    # by initialising each walker to the same age, the age never varies
    if fixed_age is not None:
        init_pars[-1] = fixed_age
        INIT_SDEV[-1] = 0.0


    # Whole emcee shebang
    sampler = emcee.EnsembleSampler(
        NWALKERS, NPAR, lnprobfunc, args=[z, star_pars]
    )
    # initialise walkers, note that INIT_SDEV is carefully chosen such that
    # all generated positions are permitted by lnprior
    pos = [
        init_pars + (np.random.random(size=len(INIT_SDEV)) - 0.5) * INIT_SDEV
        for i in range(NWALKERS)
    ]

    #N_SUCCS = 0
    #N_FAILS = 0

    # Perform burnin
    state = None
    pos, lnprob, state = sampler.run_mcmc(pos, burnin_steps, state)

    if plot_it:
        plt.clf()
        plt.plot(sampler.lnprobability)
        plt.savefig("burnin_lnprob.png")
        plt.clf()
        plt.plot(sampler.lnprobability.T)
        plt.savefig("burnin_lnprobT.png")

    #    print("Number of failed priors after burnin:\n{}".format(N_FAILS))
    #    print("Number of succeeded priors after burnin:\n{}".format(N_SUCCS))

    # Help out the struggling walkers
    best_ix = np.argmax(lnprob)
    poor_ixs = np.where(lnprob < np.percentile(lnprob, 33))
    for ix in poor_ixs:
        pos[ix] = pos[best_ix]
    sampler.reset()
    #N_FAILS = 0
    #N_SUCCS = 0

    pos, final_lnprob, rstate = sampler.run_mcmc(
        pos, sampling_steps, rstate0=state,
    )
    #    print("Number of failed priors after sampling:\n{}".format(N_FAILS))
    #    print("Number of succeeded priors after sampling:\n{}".format(N_SUCCS))
    if plot_it:
        plt.clf()
        plt.plot(sampler.lnprobability)
        plt.savefig("lnprob.png")
        plt.clf()
        plt.plot(sampler.lnprobability.T)
        plt.savefig("lnprobT.png")

    # sampler.lnprobability has shape (NWALKERS, SAMPLE_STEPS)
    # yet np.argmax takes index of flattened array
    final_best_ix = np.argmax(sampler.lnprobability)

    best_sample = sampler.flatchain[final_best_ix]

    # corner plotting is heaps funky on my laptop....
    if False:
        plt.clf()
        fig = corner.corner(sampler.flatchain, truths=best_sample)
        fig.savefig("corner.png")

    return best_sample, sampler.chain, sampler.lnprobability


def get_bayes_spreads(tb_file, z=None, plot_it=False):
    """
    For each time step, get the average spread of the stars in XYZ using Bayes

    Parameters
    ----------
    tb_file : str
        Path to a traceback file

    z [None] : [nstars] np array
        membership probability to association currently being fitted

    Output
    ------
    bayes_spreads : [ntimes] np array
        the measure of the occupied volume of a group at each time
    """
    star_pars = read_stars(tb_file)
    times = star_pars['times']
    ntimes = times.shape[0]

    bayes_spreads = np.zeros(ntimes)
    lntime_probs = np.zeros(ntimes)

    for i, time in enumerate(times):
        _, chain, lnprobability = fit_group(
            tb_file, burnin_steps=500, sampling_steps=500, z=z,
            fixed_age=time, plot_it=plot_it
        )
        bayes_spreads[i] = utils.approx_spread_from_chain(chain)
        lntime_probs[i] = np.max(lnprobability)

    lntime_probs -= np.max(lntime_probs) # shift the max to 0

    time_probs = np.exp(lntime_probs)
    return bayes_spreads, time_probs
