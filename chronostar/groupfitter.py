from __future__ import division, print_function

import numpy as np

# not sure if this is right, needed to change to this so I could run
# investigator
import astropy.constants as const
import astropy.units as u
import emcee
import logging
try:
    import matplotlib.pyplot as plt
    plt_avail = True
except ImportError:
    plt_avail = False
import pdb

import transform as tf
from _overlap import get_lnoverlaps
import traceorbit as torb
import synthesiser as syn
import datatool as dt

try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

DEFAULT_ALPHA_SIG = 1.0

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
        res -= 6 * np.log(2*np.pi)
        res -= np.log(np.linalg.det(g_cov + st_cov))
        stmg_mn = st_mn - g_mn
        stpg_cov = st_cov + g_cov
        logging.debug("ApB:\n{}".format(stpg_cov))
        res -= np.dot(stmg_mn.T, np.dot(np.linalg.inv(stpg_cov), stmg_mn))
        res *= 0.5
        lnols.append(res)
    return np.array(lnols)


def loadXYZUVW(xyzuvw_file):
    """Load mean and covariances of stars in XYZUVW space from fits file

    (If fails, auto uses getDictFromTable

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
    xyzuvw_now = fits.getdata(xyzuvw_file, 1).\
        astype('float64') #hdulist[1].data
    xyzuvw_cov_now = fits.getdata(xyzuvw_file, 2)\
        .astype('float64') #hdulist[2].data
    xyzuvw_dict = {'xyzuvw':xyzuvw_now, 'xyzuvw_cov':xyzuvw_cov_now}
    try:
        times = fits.getdata(xyzuvw_file, 3)
        xyzuvw_dict['times'] = times
    except:
        logging.info("No times in fits file")
    try:
        stars_table = fits.getdata(xyzuvw_file, 3)
        xyzuvw_dict['table'] = stars_table
    except:
        logging.info("No table in fits file")
    logging.info("Floats stored in format {}".\
                 format(xyzuvw_dict['xyzuvw'].dtype))
    return xyzuvw_dict


#def lognormal(x, mu=1.05, sig=0.105):
#    coeff = 1. / (x * sig * np.sqrt(2*np.pi))
#    expon = - (np.log(x)-mu)**2 / (2*sig**2)
#    return coeff * np.exp(expon)

def calcAlpha(dX, dV, nstars):
    """
    Assuming we have identified 100% of star mass, and that average
    star mass is 1 M_sun.

    Calculated alpha is unitless

    TODO: Astropy slows things down a very much lot. Remove it!
    """
    return ( (dV*u.km/u.s)**2 * dX*u.pc /
              (const.G * nstars * const.M_sun) ).decompose().value


def lognormal(x, mu, sig):
    """
    Caclulate lognormal parameterised by mu and sig at point x
    TODO: Actually utilise this function in lnlognormal
    """
    return 1./x * 1./(sig*np.sqrt(2*np.pi)) *\
           np.exp(-(np.log(x) - mu)**2/(2*sig**2))

def lnlognormal(x, mode=3., sig=DEFAULT_ALPHA_SIG):
    # TODO: replace lognormal innerworkings so is called with desired mode
    mu = sig**2 + np.log(mode)
    return -np.log(x*sig*np.sqrt(2*np.pi)) - (np.log(x)-mu)**2/(2*sig**2)
    # return (-np.log(x*sig*np.sqrt(2*np.pi)) -
    #         (np.log(x) - mu)**2 / (2*sig**2))

def lnAlphaPrior(pars, z, sig=DEFAULT_ALPHA_SIG):
    """
    A very approximate, gentle prior preferring super-virial distributions

    Since alpha is strictly positive, we use a lognormal prior. We then
    take the log of the result to incorporate it into the log likelihood
    evaluation.

    Mode is set at 3, when `sig` is 1, this corresponds to a FWHM of 1 dex
    (AlphaPrior(alpha=1, sig=1.) =     AlphaPrior(alpha=11,sig=1.)
                                 = 0.5*AlphaPrior(alpha=3, sig=1.)

    pars: [8] float array
        X,Y,Z,U,V,W,log(dX),log(dV),age
    star_pars:
        (retired)
    z: [nstars] float array
        membership array
    """
    dX = np.exp(pars[6])
    dV = np.exp(pars[7])
    #nstars = star_pars['xyzuvw'].shape[0]
    nstars = np.sum(z)
    alpha = calcAlpha(dX, dV, nstars)
    # taking the 10th root to make plot gentle
    # TODO: just rework mu and sig to give desired
    # mode and shape
    return lnlognormal(alpha, sig=sig)

#
# def lnlognormal(x, mu=1.05, sig=0.105):
#     # TODO: replace lognormal innerworkings so is called with desired mode
#     # mu = sigma**2 + np.log(mode)
#     # return np.log(lognormal(x, mu, sig)
#     return (-np.log(x*sig*np.sqrt(2*np.pi)) -
#             (np.log(x) - mu)**2 / (2*sig**2))


# def lnAlphaPrior(pars, star_pars, z):
#     """
#     A very approximate, gentle prior preferring super-virial distributions
#
#     Since alpha is strictly positive, we use a lognormal prior. We then
#     take the log of the result to incorporate it into the log likelihood
#     evaluation.
#
#     pars: [8] float array
#         X,Y,Z,U,V,W,log(dX),log(dV),age
#     star_pars:
#         (retired)
#     z: [nstars, ngroups] float array
#         membership array
#     """
#     dX = np.exp(pars[6])
#     dV = np.exp(pars[7])
#     #nstars = star_pars['xyzuvw'].shape[0]
#     nstars = np.sum(z)
#     alpha = calcAlpha(dX, dV, nstars)
#     # taking the 10th root to make plot gentle
#     # TODO: just rework mu and sig to give desired
#     # mode and shape
#     return lnlognormal(alpha) * 0.01


def lnprior(pars, star_pars, z):
    """Computes the prior of the group models constraining parameter space

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
    star_pars
        traceback data being fitted to
    z
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    Returns
    -------
    lnprior
        The logarithm of the prior on the model parameters

    TODO: Incorporate star determinants
    """
    # fetch maximum allowed age
    max_age = 500

    means = pars[0:6]
    stds = np.exp(pars[6:8])
    age = pars[8]

    if np.min(means) < -100000 or np.max(means) > 100000:
        return -np.inf
    if np.min(stds) <= 0.0 or np.max(stds) > 10000.0:
        return -np.inf
    if age < 0.0 or age > max_age:
        return -np.inf

    return lnAlphaPrior(pars, z, sig=DEFAULT_ALPHA_SIG)


def getLogOverlaps(pars, star_pars):
    """
    Given the parametric description of an origin, calculate star overlaps

    Utilises Overlap, a c module wrapped with swig to be callable by python.
    This allows a 100x speed up in our 6x6 matrix operations when compared
    to numpy.

    Parameters
    ----------
    pars: [npars] list
        Parameters describing the origin of group
        typically [X,Y,Z,U,V,W,np.log(dX),np.log(dV),age]
    star_pars: dict
        traceback data being fitted to, stored as a dict:
        'xyzuvw': [nstars,6] float array
            the central estimates of each star in XYZUVW space
        'xyzuvw_cov': [nstars,6,6] float array
            the covariance of each star in XYZUVW space
    """
    # convert pars into covariance matrix with our Group class
    g = syn.Group(pars, internal=True)
    cov_then = g.generateCovMatrix()

    # Trace group pars forward to now
    mean_now = torb.traceOrbitXYZUVW(g.mean, g.age, True)
    cov_now = tf.transform_cov(
        cov_then, torb.traceOrbitXYZUVW, g.mean, dim=6, args=(g.age, True)
    )

    nstars = star_pars['xyzuvw'].shape[0]
    lnols = get_lnoverlaps(
        cov_now, mean_now, star_pars['xyzuvw_cov'], star_pars['xyzuvw'],
        nstars
    )
    return lnols


def lnlike(pars, star_pars, z):
    """Computes the log-likelihood for a fit to a group.

    The emcee parameters encode the modelled origin point of the stars.
    Using the parameters, a mean and covariance in 6D space are constructed
    as well as an age. The kinematics are then projected forward to the
    current age and compared with the current stars' XYZUVW values (and
    uncertainties)

    P(D|G) = \prod_i[P(d_i|G)^{z_i}]
    \ln P(D|G) = \sum_i z_i*\ln P(d_i|G)

    Parameters
    ----------
    pars: [npars] list
        Parameters describing the group model being fitted
    star_pars: dict
        traceback data being fitted to, stored as a dict:
        'xyzuvw': [nstars,6] float array
            the central estimates of each star in XYZUVW space
        'xyzuvw_cov': [nstars,6,6] float array
            the covariance of each star in XYZUVW space
    z: [nstars] float array
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    Returns
    -------
    lnlike
        the logarithm of the likelihood of the fit
    """
    lnols = getLogOverlaps(pars, star_pars)
    if np.isnan(lnols).any():
        pdb.set_trace()
    if (np.sum(lnols*z)) is np.nan:
        pdb.set_trace()
    return np.sum(lnols * z)


def lnprobFunc(pars, star_pars, z):
    """Computes the log-probability for a fit to a group.

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
        0,1,2,3,4,5,   6,   7,  8
        X,Y,Z,U,V,W,lndX,lndV,age
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
    lp = lnprior(pars, star_pars, z)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(pars, star_pars, z)


def lnpriorTraceback(pars, star_pars, z):
    """Computes the prior of the group models constraining parameter space

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
    star_pars
        traceback data being fitted to
    z
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    Returns
    -------
    lnprior
        The logarithm of the prior on the model parameters

    TODO: Incorporate star determinants
    """
    # fetch maximum allowed age
    max_age = abs(star_pars['times'][-1])

    means = pars[0:6]
    stds = np.exp(pars[6:8])
    age = pars[8]

    if np.min(means) < -100000 or np.max(means) > 100000:
        return -np.inf
    if np.min(stds) <= 0.0 or np.max(stds) > 10000.0:
        return -np.inf
    if age < 0.0 or age >= max_age:
        return -np.inf

    return lnAlphaPrior(pars, z)


def interp_cov(target_time, star_pars):
    """Calculates the xyzuvw vector and covariance matrix by interpolation

    Parameters
    ---------
    target_time
        The desired time to be fitted to (should be positive)
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
    times = abs(star_pars['times'])
    ix = np.interp(target_time, times, np.arange(len(times)))
    ## check if interpolation is necessary
    # if np.isclose(target_time, times, atol=1e-5).any():
    #    ix0 = int(round(ix))
    #    interp_mns = star_pars['xyzuvw'][:, ix0]
    #    interp_covs = star_pars['xyzuvw_cov'][:, ix0]
    #    return interp_covs, interp_mns

    ix0 = np.int(ix)
    frac = ix - ix0
    interp_covs = star_pars['xyzuvw_cov'][:, ix0] * (1 - frac) + \
                  star_pars['xyzuvw_cov'][:, ix0 + 1] * frac

    interp_mns = star_pars['xyzuvw'][:, ix0] * (1 - frac) + \
                 star_pars['xyzuvw'][:, ix0 + 1] * frac

    return interp_covs, interp_mns


def lnlikeTraceback(pars, star_pars, z, return_lnols=False):
    """
    Similar to lnlike, but for the (retired) traceback implementation.

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
    star_pars
        traceback data being fitted to
    z
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

    # Interpolate star means and covs to the groups age
    interp_covs, interp_mns = interp_cov(age, star_pars)

    nstars = star_pars['xyzuvw'].shape[0]
    lnols = get_lnoverlaps(
        cov_then, mean_then, interp_covs, interp_mns,
        nstars
    )

    if return_lnols:
        return lnols

    return np.sum(lnols * z)


def lnprobTracebackFunc(pars, star_pars, z):
    """
    Similar to lnprobFunc, but for the (retired) traceback implementation

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
        0,1,2,3,4,5,   6,   7,  8
        X,Y,Z,U,V,W,lndX,lndV,age
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
    lp = lnpriorTraceback(pars, z)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikeTraceback(pars, star_pars, z)


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


def fitGroup(xyzuvw_dict=None, xyzuvw_file='', z=None, burnin_steps=1000,
             plot_it=False, pool=None, convergence_tol=0.25, init_pos=None,
             plot_dir='', save_dir='', init_pars=None, sampling_steps=None,
             traceback=False, max_iter=None, init_at_mean=False):
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
    if xyzuvw_dict is None:
        star_pars = dt.loadXYZUVW(xyzuvw_file)
    else:
        star_pars = xyzuvw_dict

    #            X,Y,Z,U,V,W,lndX,lndV,age
    INIT_SDEV = [20,20,20,5,5,5, 0.5, 0.5,1]

    # initialise z if needed as array of 1s of length nstars
    if z is None:
        z = np.ones(star_pars['xyzuvw'].shape[0])

    # Initialise the fit
    if init_pars is None:
        init_pars = np.array([0,0,0,0,0,0,3.,2.,3.0])
        if init_at_mean:
            init_pars[:6] = np.mean(star_pars['xyzuvw'], axis=0)


    NPAR = len(init_pars)
    NWALKERS = 2 * NPAR

    # Since emcee linearly interpolates between walkers to determine next
    # step by initialising each walker to the same age, the age never
    # varies
#    if fixed_age is not None:
#        init_pars[-1] = fixed_age
#        INIT_SDEV[-1] = 0.0

    if traceback:
        # trace stars back in time
        lnprob_function = lnprobTracebackFunc
    else:
        # project group forward through time
        lnprob_function = lnprobFunc

    # Whole emcee shebang
    sampler = emcee.EnsembleSampler(
        NWALKERS, NPAR, lnprob_function, args=[star_pars, z], pool=pool,
    )
    # initialise walkers, note that INIT_SDEV is carefully chosen such thata
    # all generated positions are permitted by lnprior
    if init_pos is None:
        logging.info("Initialising walkers around the parameters:\n{}".\
                     format(init_pars))
        logging.info("With standard deviation:\n{}".\
                     format(INIT_SDEV) )
        pos = np.array([
            init_pars + (np.random.random(size=len(INIT_SDEV)) - 0.5)\
            * INIT_SDEV
            for _ in range(NWALKERS)
        ])
        # force ages to be positive
        pos[:,-1] = abs(pos[:,-1])
    else:
        pos = np.array(init_pos)
        logging.info("Using provided positions which have mean:\n{}".\
                     format(np.mean(pos, axis=0)))

    # Perform burnin
    state = None
    converged = False
    cnt = 0
    logging.info("Beginning burnin loop")
    burnin_lnprob_res = np.zeros((NWALKERS,0))

    # burn in until converged or the optional max_iter is reached
    while (not converged) and cnt != max_iter:
        logging.info("Burning in cnt: {}".format(cnt))
        sampler.reset()
        pos, lnprob, state = sampler.run_mcmc(pos, burnin_steps, state)
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
                pos[ix] = pos[best_ix]

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
        pos, lnprob, state = sampler.run_mcmc(pos, sampling_steps, state)
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
#        logging.info("Plotting corner")
#        plt.clf()
#        corner.corner(sampler.flatchain)
#        plt.savefig(plot_dir+"corner.pdf")
        logging.info("Plotting done")

    # sampler.lnprobability has shape (NWALKERS, SAMPLE_STEPS)
    # yet np.argmax takes index of flattened array
    final_best_ix = np.argmax(sampler.lnprobability)
    best_sample = sampler.flatchain[final_best_ix]

    # displaying median and range of each paramter
    med_and_span = dt.calcMedAndSpan(sampler.chain)
    logging.info("Results:\n{}".format(med_and_span))

    return best_sample, sampler.chain, sampler.lnprobability


