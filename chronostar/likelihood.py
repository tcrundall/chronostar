"""
A module containing all the required functions to evaluate the likelhood
of a Component model and data.

TODO: Implement (and place sensibly) means to generate covariance matrices
from astropy table
"""
import logging
import numpy as np

from component import Component
USE_C_IMPLEMENTATION = True
try:
    from _overlap import get_lnoverlaps
except ImportError:
    USE_C_IMPLEMENTATION = False

def slowGetLogOverlaps(g_cov, g_mn, st_covs, st_mns):
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

    Returns
    -------
    ln_ols : ([nstars] float array)
        an array of the logarithm of the overlaps
    """
    lnols = []
    for st_cov, st_mn in zip(st_covs, st_mns):
        res = 0
        res -= 6 * np.log(2*np.pi)
        res -= np.log(np.linalg.det(g_cov + st_cov))
        stmg_mn = st_mn - g_mn
        stpg_cov = st_cov + g_cov
        logging.debug("ApB:\n{}".format(stpg_cov))
        res -= np.dot(stmg_mn.T, np.dot(np.linalg.inv(stpg_cov), stmg_mn))
        res *= 0.5
        lnols.append(res)
    return np.array(lnols)


def calcAlpha(dx, dv, nstars):
    """
    Assuming we have identified 100% of star mass, and that average
    star mass is 1 M_sun.

    Calculated alpha is unitless

    TODO: Astropy slows things down a very much lot. Remove it!
    """
    # G_const taken from astropy
    G_const = 4.30211e-3 #pc (km/s)^2 / M_sol
    M_sol = 1. # M_sol
    return (dv**2 * dx) / (G_const * nstars * M_sol)


def lnlognormal(x, mu=2.1, sig=1.0):
    return -np.log(x*sig*np.sqrt(2*np.pi)) - (np.log(x)-mu)**2/(2*sig**2)


def lnAlphaPrior(comp, memb_probs, sig=1.0):
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
    data:
        (retired)
    memb_probs: [nstars] float array
        membership array
    """
    dX = comp.sphere_dx
    dV = comp.sphere_dv
    nstars = np.sum(memb_probs)
    alpha = calcAlpha(dX, dV, nstars)
    return lnlognormal(alpha, mu=2.1, sig=sig)


def lnprior(comp, memb_probs):
    """Computes the prior of the group models constraining parameter space

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
    data
        traceback data being fitted to
    memb_probs
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    Returns
    -------
    lnprior
        The logarithm of the prior on the model parameters
    """
    # fetch maximum allowed age
    max_age = 500

    mean = comp.mean
    stds = np.linalg.eigvalsh(comp.covmatrix)
    if np.min(comp.mean) < -100000 or np.max(comp.mean) > 100000:
        return -np.inf
    if np.min(stds) <= 0.0 or np.max(stds) > 10000.0:
        return -np.inf
    if comp.age < 0.0 or comp.age > max_age:
        return -np.inf

    return lnAlphaPrior(comp, memb_probs, sig=1.0)


def getLogOverlaps(comp, data):
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
    data: dict
        traceback data being fitted to, stored as a dict:
        'xyzuvw': [nstars,6] float array
            the central estimates of each star in XYZUVW space
        'xyzuvw_cov': [nstars,6,6] float array
            the covariance of each star in XYZUVW space
    """
    # Prepare star arrays
    cov_stars = buildStarCovs(data)
    mean_stars = buildStarMeans(data)
    nearby_star_count = len(mean_stars)

    # Get current day projection of component
    cov_now, mean_now = comp.getCurrentDayProjection()

    # Calculate overlap integral of each star
    if USE_C_IMPLEMENTATION:
        lnols = get_lnoverlaps(cov_now, mean_now, cov_stars, mean_stars,
                           nearby_star_count)
    else:
        lnols = slowGetLogOverlaps(cov_now, mean_now, cov_stars, mean_stars)
    return lnols


def lnlike(comp, data, memb_probs):
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
    data: dict
        traceback data being fitted to, stored as a dict:
        'xyzuvw': [nstars,6] float array
            the central estimates of each star in XYZUVW space
        'xyzuvw_cov': [nstars,6,6] float array
            the covariance of each star in XYZUVW space
    memb_probs: [nstars] float array
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    Returns
    -------
    lnlike
        the logarithm of the likelihood of the fit
    """
    # Only consider contributions of stars with larger than 0.1% membership prob
    memb_threshold = 0.001
    nearby_star_mask = np.where(memb_probs > memb_threshold)

    # Calculate log overlaps of relevant stars
    lnols = np.zeros(len(memb_probs))
    lnols[nearby_star_mask] = getLogOverlaps(comp, data[nearby_star_mask])

    # Weight each stars contribution by their membership probability
    result = np.sum(lnols * memb_probs)
    return result


def lnprobFunc(pars, data, memb_probs, form='sphere'):
    """Computes the log-probability for a fit to a group.

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
        0,1,2,3,4,5,   6,   7,  8
        X,Y,Z,U,V,W,lndX,lndV,age
    data
        data being fitted to
    memb_probs
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    Returns
    -------
    logprob
        the logarithm of the posterior probability of the fit
    """
    comp = Component(pars, form=form, internal=True)
    lp = lnprior(comp, memb_probs)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(comp, data, memb_probs)
