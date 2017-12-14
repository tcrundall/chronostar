from __future__ import division, print_function

import numpy as np
import sys

from chronostar._overlap import get_lnoverlaps
import corner
import emcee
import matplotlib.pyplot as plt
import pdb
import pickle
from utils import generate_cov
try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits

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
    if len(tb_file)==0:
        print("Input a filename...")
        raise UserWarning

    #Stars is an astropy.Table of stars
    if tb_file[-3:] == 'pkl':
        with open(tb_file,'r') as fp:
            (stars,times,xyzuvw,xyzuvw_cov)=pickle.load(fp)
    elif (tb_file[-3:] == 'fit') or (tb_file[-4:] == 'fits'):
        stars = pyfits.getdata(tb_file,1)
        times = pyfits.getdata(tb_file,2)
        xyzuvw = pyfits.getdata(tb_file,3)
        xyzuvw_cov = pyfits.getdata(tb_file,4)
    else:
        print("Unknown File Type!")
        raise UserWarning

    return dict(stars=stars,times=times,xyzuvw=xyzuvw,xyzuvw_cov=xyzuvw_cov)

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
    ix0 = np.int(ix)
    frac = ix-ix0
    interp_covs     = star_pars['xyzuvw_cov'][:,ix0]*(1-frac) +\
                      star_pars['xyzuvw_cov'][:,ix0+1]*frac

    interp_mns       = star_pars['xyzuvw'][:,ix0]*(1-frac) +\
                       star_pars['xyzuvw'][:,ix0+1]*frac

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
    group_cov = generate_cov(pars)

    # check if covariance matrix is singular
    if np.min( np.linalg.eigvalsh(group_cov) ) < 0:
        return -np.inf

    group_mn   = pars[0:6]

    # interpolate star data to modelled age
    age = pars[13]
    interp_covs, interp_mns = interp_cov(age, star_pars)
    nstars = interp_mns.shape[0]
    
    # PROGRESS HALTED! REWRITE C CODE IN LOGARITHMS BEFORE CONTINUING
    lnols = get_lnoverlaps(
        group_cov, group_mn, interp_covs, interp_mns, nstars
    )
    
    return np.sum(z*lnols)

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
    global N_FAILS
    global N_SUCCS
    lp = lnprior(pars, z, star_pars)
    if not np.isfinite(lp):
        N_FAILS += 1
        return -np.inf
    N_SUCCS += 1
    return lp + lnlike(pars, z, star_pars)

def fit_group(tb_file, z=None, init_pars=None, plot_it=False):
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
    
    Returns
    -------
    best_sample
        The parameters of the group model which yielded the highest posterior
        probability
    """
    global N_FAILS
    global N_SUCCS
    # initialise some emcee constants
    BURNIN_STEPS = 500
    SAMPLING_STEPS = 1000
    if init_pars is None:
        #            X,Y,Z,U,V,W,1/dX,1/dY,1/dZ,1/dV,Cxy,Cxz,Cyz,age
        init_pars = [0,0,0,0,0,0, 0.1, 0.1, 0.1, 0.2,0.0,0.0,0.0,2.0]
    NPAR = len(init_pars)
    NWALKERS = 2*NPAR
    #            X,Y,Z,U,V,W,1/dX,1/dY,1/dZ,1/dV,Cxy,Cxz,Cyz,age
    INIT_SDEV = [1,1,1,1,1,1,0.01,0.01,0.01,0.02,0.1,0.1,0.1,0.5]
    star_pars = read_stars(tb_file)

    # initialise z if needed as array of 1s of length nstars
    if z is None:
        z = np.ones(star_pars['xyzuvw'].shape[0])

    # Whole emcee shebang
    sampler = emcee.EnsembleSampler(
        NWALKERS, NPAR, lnprobfunc, args=[z, star_pars]
    )
    # initialise walkers, note that INIT_SDEV is carefully chosen such that
    # all generated positions are permitted by lnprior
    pos = [
        init_pars + (np.random.random(size=len(INIT_SDEV)) - 0.5)*INIT_SDEV
        for i in range(NWALKERS)
    ]

    N_SUCCS = 0 
    N_FAILS = 0
        
    # Perform burnin
    state = None
    pos, lnprob, state = sampler.run_mcmc(pos, BURNIN_STEPS, state)

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
    N_FAILS = 0
    N_SUCCS = 0 

    pos, final_lnprob, rstate = sampler.run_mcmc(
        pos, SAMPLING_STEPS, rstate0=state,
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

    return best_sample

