from __future__ import division, print_function

import numpy as np

# not sure if this is right, needed to change to this so I could run
# investigator
import emcee
import logging
import matplotlib.pyplot as plt
import pickle

import transform as tf
from _overlap import get_lnoverlaps
import traceorbit as torb
import synthesiser as syn

try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

def slowGetLogOverlaps(g_cov, g_mn, st_covs, st_mns, nstars):
    """
    A pythonic implementation of overlap integral calculation

    Paramters
    ---------
    g_cov : ([6,6] float array)
        Covariance matrix of the group
    g_mn : ([6] float array)
        mean of the group
    st_covs : ([nstars, 6, 6] float array)
        covariance matrices of the stars
    st_mns : ([nstars, 6], float array)
        means of the stars
    nstars : int
        number of stars

    Returns
    -------
    ln_ols : ([nstars] float array)
        an array of the logarithm of the overlaps
    """
    lnols = []
    for st_cov, st_mn in zip(st_covs, st_mns):
        res = 0
        #res += 6 * np.log(2*np.pi)
        res -= 3 * np.log(2*np.pi)
        res -= 0.5 * np.log(np.linalg.det(g_cov + st_cov))
        stmg_mn = st_mn - g_mn
        stpg_cov = st_cov + g_cov
        logging.debug("ApB:\n{}".format(stpg_cov))
        res -= 0.5 * np.dot(stmg_mn.T, np.dot(np.linalg.inv(stpg_cov), stmg_mn))
        lnols.append(res)
    return np.array(lnols)


def loadXYZUVW(xyzuvw_file):
    """Load mean and covariances of stars in XYZUVW space from fits file

    Parameters
    ----------
    xyzuvw_file : (String)
        Ideally *.fits, the file name of the fits file with and hdulist:
            [1] : xyzuvw
            [2] : xyzuvw covariances

    Returns
    -------
    xyzuvw_dict : (dictionary)
        xyzuvw : ([nstars, 6] float array)
            the means of the stars
        xyzuvw_cov : ([nstars, 6, 6] float array)
            the covariance matrices of the stars
    """
    if (xyzuvw_file[-3:] != 'fit') and (xyzuvw_file[-4:] != 'fits'):
        xyzuvw_file = xyzuvw_file + ".fits"
    # TODO Ask Mike re storing fits files as float64 (instead of '>f8')
    xyzuvw_now = fits.getdata(xyzuvw_file, 1).astype('float64') #hdulist[1].data
    xyzuvw_cov_now = fits.getdata(xyzuvw_file, 2).astype('float64') #hdulist[2].data
    xyzuvw_dict = {'xyzuvw':xyzuvw_now, 'xyzuvw_cov':xyzuvw_cov_now}
    try:
        stars_table = fits.getdata(xyzuvw_file, 3)
        xyzuvw_dict['table'] = stars_table
    except:
        pass
    logging.info("Floats stored in format {}".\
                 format(xyzuvw_dict['xyzuvw'].dtype))
    return xyzuvw_dict


def lnprior(pars, star_pars):
    """Computes the prior of the group models constraining parameter space

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
    star_pars
        traceback data being fitted to
    (retired) z
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    Returns
    -------
    lnprior
        The logarithm of the prior on the model parameters

    TODO: Incorporate star determinants
    """
    # fetch maximum allowed age
    max_age = 200

    means = pars[0:6]
    stds = np.exp(pars[6:8])
    age = pars[8]

    if np.min(means) < -1000 or np.max(means) > 1000:
        return -np.inf
    if np.min(stds) <= 0.0 or np.max(stds) > 1000.0:
        return -np.inf
    if age < 0.0 or age > max_age:
        return -np.inf
    return 0.0


def lnlike(pars, star_pars, z=None, return_lnols=False):
    """Computes the log-likelihood for a fit to a group.

    The emcee parameters encode the modelled origin point of the stars. Using
    the parameters, a mean and covariance in 6D space are constructed as well as
    an age. The kinematics are then projected forward to the current age and
    compared with the current stars' XYZUVW values (and uncertainties)

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
    star_pars
        traceback data being fitted to
    (retired) z
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    Returns
    -------
    lnlike
        the logarithm of the likelihood of the fit
    """
    # convert pars into covariance matrix
    g = syn.Group(pars, internal=True)
    mean_then = g.mean
    cov_then = g.generateCovMatrix()
    age = g.age

    # Trace group pars forward to now
    mean_now = torb.traceOrbitXYZUVW(g.mean, g.age, True)
    cov_now = tf.transform_cov(
        cov_then, torb.traceOrbitXYZUVW, g.mean, dim=6, args=(g.age, True)
    )

    nstars = star_pars['xyzuvw'].shape[0]
    lnols = get_lnoverlaps(
        cov_now, mean_now, star_pars['xyzuvw_cov'], star_pars['xyzuvw'], nstars
    )
    if return_lnols:
        return lnols

    return np.sum(lnols * z)


def lnprobFunc(pars, star_pars, z):
    """Computes the log-probability for a fit to a group.

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
        0,1,2,3,4,5, 6, 7,  8
        X,Y,Z,U,V,W,dX,dV,age
    star_pars
        traceback data being fitted to
    (retired) z
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    Returns
    -------
    logprob
        the logarithm of the posterior probability of the fit
    """
    lp = lnprior(pars, star_pars)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(pars, star_pars, z)

def burninConvergence(lnprob, tol=0.1, slice_size=100, cutoff=0):
    """Checks early lnprob vals with final lnprob vals for convergence

    Parameters
    ----------
    lnprob : [nwalkers, nsteps] array

    tol : float
        The number of standard deviations the final mean lnprob should be within
        of the initial mean lnprob

    slice_size : int
        Number of steps at each end to use for mean lnprob calcultions

    cuttoff : int
        Step number at which to start the analysis. i.e., a value of 0 would
        mean to include the whole chain, whereas a value of 500 would be to
        only confirm no deviation in the steps [500:END]
    """
    # take a chunk the smaller of 100 or half the chain
    if lnprob.shape[1] < slice_size:
        slice_size = int(round(0.5*lnprob.shape[1]))

    start_lnprob_mn = np.mean(lnprob[:,:slice_size])
    start_lnprob_std = np.std(lnprob[:,:slice_size])

    end_lnprob_mn = np.mean(lnprob[:, -slice_size:])
    end_lnprob_std = np.std(lnprob[:, -slice_size:])

    return np.isclose(start_lnprob_mn, end_lnprob_mn, atol=tol*end_lnprob_std)

def fitGroup(xyzuvw_dict=None, xyzuvw_file='', z=None, burnin_steps=1000,
             plot_it=False, pool=None, convergence_tol=0.1, init_pos=None,
             plot_dir='', save_dir='', init_pars=None, sampling_steps=None):
    """Fits a single gaussian to a weighted set of traceback orbits.

    Parameters
    ----------
    xyzuvw_dict, xyzuvw_file : (dict or string)
        Can either pass in the dictionary directly, or a '.fits' filename from
        which to load the dictionary:
            xyzuvw (nstars,ntimes,6) numpy array, XYZ in pc and UVW in km/s
            xyzuvw_cov (nstars,ntimes,6,6) numpy array, covariance of xyzuvw
            table: (nstars) high astropy table including columns as
                        documented in the measurer class.
    z : ([nstars] array {None})
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    burnin_steps : int {1000}

    plot_it : bool {False}

    pool : MPIPool object {None}
        pool of threads to execute walker steps concurrently

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
    if xyzuvw_dict is None:
        star_pars = loadXYZUVW(xyzuvw_file)
    else:
        star_pars = xyzuvw_dict

    #            X,Y,Z,U,V,W,lndX,lndV,age
    INIT_SDEV = [20,20,20,5,5,5, 0.5, 0.5,1]

    # initialise z if needed as array of 1s of length nstars
    if z is None:
        z = np.ones(star_pars['xyzuvw'].shape[0])

    # Initialise the fit
    if init_pars is None:
        init_pars = [0,0,0,0,0,0,3.,2.,10.0]

    NPAR = len(init_pars)
    NWALKERS = 2 * NPAR

    # Since emcee linearly interpolates between walkers to determine next step
    # by initialising each walker to the same age, the age never varies
#    if fixed_age is not None:
#        init_pars[-1] = fixed_age
#        INIT_SDEV[-1] = 0.0

    # Whole emcee shebang
    sampler = emcee.EnsembleSampler(
        NWALKERS, NPAR, lnprobFunc, args=[star_pars, z], pool=pool,
    )
    # initialise walkers, note that INIT_SDEV is carefully chosen such that
    # all generated positions are permitted by lnprior
    if init_pos is None:
        pos = [
            init_pars + (np.random.random(size=len(INIT_SDEV)) - 0.5) * INIT_SDEV
            for _ in range(NWALKERS)
        ]
    else:
        pos = init_pos

    # Perform burnin
    state = None
    converged = False
    cnt = 0
    logging.info("Beginning burnin loop")
    burnin_lnprob_res = np.zeros((NWALKERS,0))
    while not converged:
        logging.info("Burning in cnt: {}".format(cnt))
        sampler.reset()
        pos, lnprob, state = sampler.run_mcmc(pos, burnin_steps, state)
        converged = burninConvergence(sampler.lnprobability, tol=convergence_tol)

        # Help out the struggling walkers
        best_ix = np.argmax(lnprob)
        poor_ixs = np.where(lnprob < np.percentile(lnprob, 33))
        for ix in poor_ixs:
            pos[ix] = pos[best_ix]

        if plot_it:
            plt.clf()
            plt.plot(sampler.lnprobability.T)
            plt.savefig(plot_dir+"burnin_lnprobT{:02}.png".format(cnt))

        burnin_lnprob_res = np.hstack((
            burnin_lnprob_res, sampler.lnprobability
        ))
        cnt += 1

    logging.info("Burnt in, with convergence: {}".format(converged))
    if plot_it:
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
        pos, lnprob, state = sampler.run_mcmc(pos, sampling_steps, state)
        logging.info("Sampling done")

    # save the chain for later inspection
    np.save(save_dir+"final_chain.npy", sampler.chain)
    np.save(save_dir+"final_lnprob.npy", sampler.lnprobability)

#    print("Sampled")
    if plot_it:
        plt.clf()
        plt.plot(sampler.lnprobability.T)
        plt.savefig(plot_dir+"lnprobT.png")

    # sampler.lnprobability has shape (NWALKERS, SAMPLE_STEPS)
    # yet np.argmax takes index of flattened array
    final_best_ix = np.argmax(sampler.lnprobability)
    best_sample = sampler.flatchain[final_best_ix]

    return best_sample, sampler.chain, sampler.lnprobability
