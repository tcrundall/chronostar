from __future__ import division, print_function

import numpy as np
import sys

def read_stars(tb_file):
    """Read stars from traceback file into a dictionary.

    Notes
    -----
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
        xyzuvw (nstars,ntimes,6) numpy array, XYZ in pc and UVW in km/s
        xyzuvw_cov (nstars,ntimes,6,6) numpy array, covariance of xyzuvw
    """
    return 0

def generate_icov(pars):
    """Generate inverse covariance matrix from standard devs and correlations

    Parameters
    ----------
    pars
        [14] array with the following values:
        pars[6:10] : [1/dX, 1/dY, 1/dZ, 1/dV] :
            standard deviations in position and velocity
            for group model or stellar PDFs
        pars[10:13] : [CorrXY, CorrXZ, CorrYZ]
            correlations between position

    Returns
    -------
    icov
        [6, 6] array : incovariance matrix for group model or stellar pdf
    """
    dX, dY, dZ, dV = 1.0/np.array(pars[6:10])
    Cxy, Cxz, Cyz  = pars[10:13]
    cov = np.array([
            [dX**2,     Cxy*dX*dY, Cxz*dX*dZ, 0.0,   0.0,   0.0],
            [Cxy*dX*dY, dY**2,     Cyz*dY*dZ, 0.0,   0.0,   0.0],
            [Cxz*dX*dZ, Cyz*dY*dZ, dZ**2,     0.0,   0.0,   0.0],
            [0.0,       0.0,       0.0,       dV**2, 0.0,   0.0],
            [0.0,       0.0,       0.0,       0.0,   dV**2, 0.0],
            [0.0,       0.0,       0.0,       0.0,   0.0,   dV**2],
        ])
    icov = np.linalg.inv(cov)
    return icov

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
    interp_mns
        [nstars, 6] array with phase values for each star at interpolated
        time
    interp_covs
        [nstars, 6, 6] array with covariance matrix of the phase values
        for each star at interpolated time
    """
    times = star_pars['times']
    ix = np.interp(target_time, times, np.arange(len(times)))
    ix0 = np.int(ix)
    frac = ix-ix0
    interp_mns       = star_pars['xyzuvw'][:,ix0]*(1-frac) +\
                       star_pars['xyzuvw'][:,ix0+1]*frac

    interp_covs     = star_pars['xyzuvw_cov'][:,ix0]*(1-frac) +\
                      star_pars['xyzuvw_cov'][:,ix0+1]*frac

    return interp_mns, interp_covs

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
    #eig_val = 1 / inv_eig_val
    prior = eig_val / (char_min**(2) + eig_val**2)
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

    means  = pars[0:6]
    inv_stds  = pars[6:10]
    corrs = pars[10:13]
    age   = pars[13]

    if np.min(means) < -1000 or np.max(means) > 1000:
        return -np.inf
    if np.min(inv_stds) <= 0.0 or np.max(inv_stds) > 100.0:
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
    group_icov = generate_icov(pars)
    group_mn   = pars[0:6]

    # interpolate star data to modelled age
    age = pars[13]
    interp_mns, interp_covs = interp_cov(age, star_pars)
    
    # PROGRESS HALTED! REWRITE C CODE IN LOGARITHMS BEFORE CONTINUING

    return 0

def lnprob(pars, z, star_pars):
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

    return 0

def fit_group(tb_file, z=None):
    """Fits a single gaussian to a weighted set of traceback orbits.

    Parameters
    ----------
    tb_file
        a '.pkl' or '.fits' file containing traceback orbits
    z
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.
    
    Returns
    -------
    best_fit
        The parameters of the group model which yielded the highest posterior
        probability
    """
    return 0
