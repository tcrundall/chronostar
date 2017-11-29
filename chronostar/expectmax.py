"""
a module for implementing the expectation-maximisation algorithm
in order to fit a multi-gaussian mixture model of moving groups' origins
to a data set of stars tracedback through XYZUVW

todo:
    - implement average error cacluation in lnprobfunc
"""
from __future__ import print_function, division

import emcee        # ... duh
import sys          # for MPI
import numpy as np
import matplotlib.pyplot as plt
import pdb          # for debugging
import corner       # for pretty corner plots
import pickle       # for dumping and reading data
# for permuting samples when realigning
from sympy.utilities.iterables import multiset_permutations 
try:
    import astropy.io.fits as pyfits
except:
    import pyfits

try:
    import _overlap as overlap #&TC
except:
    print("overlap not imported, SWIG not possible. Need to make in directory...")
from emcee.utils import MPIPool

try:                # don't know why we use xrange to initialise walkers
    xrange
except NameError:
    xrange = range

def read_stars(infile):
    """
    Read stars from a previous pickle file into a dictionary.
    
    The input is an error ellipse in 6D (X,Y,Z,U,V,W) of a list of stars, at
    a bunch of times in the past.
    
    Parameters
    ----------
    infile: string
        input pickle file
        
    Returns
    -------
    star_dictionary: dict
        stars: (nstars) high astropy table including columns as 
                    documented in the Traceback class.
        times: (ntimes) numpy array, containing times that have 
                    been traced back, in Myr
        xyzuvw (nstars,ntimes,6) numpy array, XYZ in pc and UVW in km/s
        xyzuvw_cov (nstars,ntimes,6,6) numpy array, covariance of xyzuvw
    """
    if len(infile)==0:
        print("Input a filename...")
        raise UserWarning

    #Stars is an astropy.Table of stars
    if infile[-3:] == 'pkl':
        with open(infile,'r') as fp:
            (stars,times,xyzuvw,xyzuvw_cov)=pickle.load(fp)
    elif (infile[-3:] == 'fit') or (infile[-4:] == 'fits'):
        stars = pyfits.getdata(infile,1)
        times = pyfits.getdata(infile,2)
        xyzuvw = pyfits.getdata(infile,3)
        xyzuvw_cov = pyfits.getdata(infile,4)
    else:
        print("Unknown File Type!")
        raise UserWarning
    xyzuvw0 = xyzuvw[:][0]

    return dict(stars=stars,times=times,xyzuvw=xyzuvw,xyzuvw_cov=xyzuvw_cov)
                   #xyzuvw_icov=xyzuvw_icov,xyzuvw_icov_det=xyzuvw_icov_det)

def generate_cov(group_pars):
    """
    Generate the covariance matrix described by the group parameters.
    ------
    Input:
        group_pars : [14] array,
            [X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars]
    Output:
        cov : [6, 6] array, covariance matrix of group
    """
    cov = np.eye(6)
    cov[np.tril_indices(3,-1)] = group_pars[10:13]
    cov[np.triu_indices(3,1)] = group_pars[10:13]
    #Note that 'pars' stores the inverse of the standard deviation
    #Convert correlation to covariance for position.
    for i in range(3):
        cov[i,:3] *= 1 / group_pars[6:9]
        cov[:3,i] *= 1 / group_pars[6:9]
    #Convert correlation to covariance for velocity.
    for i in range(3,6):
        cov[i,3:] *= 1 / group_pars[9]
        cov[3:,i] *= 1 / group_pars[9]
    
    return cov

def interp_cov(target_time, star_params):
    """
    Interpolate in time to get a xyzuvw vector and covariance matrix. This
    is used to find the kinematics of stars between the discrete traceback
    instances.
    -----
    Input:
        target_time : float
            the desired time
        star_params : dict
            keys: {'table...?', 'xyzuvw', 'xyzuvw_cov', 'times'}
    Output:
        xyzuvw : [nstar, 6] array, phase space vector for all stars
        xyzuvw_cov : [nstar, 6, 6] array, covaraince matrix for all stars
    """
    times = star_params['times']
    ix = np.interp(target_time, times, np.arange(len(times)))
    ix0 = np.int(ix)
    frac = ix-ix0
    interp_mns       = star_params['xyzuvw'][:,ix0]*(1-frac) +\
                       star_params['xyzuvw'][:,ix0+1]*frac

    interp_covs     = star_params['xyzuvw_cov'][:,ix0]*(1-frac) +\
                      star_params['xyzuvw_cov'][:,ix0+1]*frac

    return interp_mns, interp_covs

def ln_eig_prior(char_min_pos, char_min_vel, eig_vals):
    """
    Calculates a prior based on the eigen values (i.e. lengths of the major
    and minor axes of the ellipsoid defined by the covariance matrix of the
    group model).
    
    Remember to log this!!
    ------
    Input:
        char_min_pos : float, the characteristic minimum of position dispersion
        char_min_vel : float, the characterisitc minimum of velocity dispersion
        eig_vals : [4] array (float), 3 position e.v.s and 1 velocity e.v
    Output:
        ln_eig_prior : float
    """
    eig_prior = 1.0
    for pos_eig_val in eig_vals[0:3]:
        eig_prior *= pos_eig_val / (char_min_pos**2 + pos_eig_val**2)
    eig_prior *= eig_vals[3] / (char_min_vel**2 + eig_vals[3]**2)
    return np.log(eig_prior)

def lnprior(group_pars, star_params):
    """
    Takes the parameters of a group and generates the log prior. This is also
    used to enforce hard limits e.g. to keep dX > 0.
    This function will also be used to incorporate priors around the data set
    in order to retrieve the true origins of the group. These priors will
    be calibrated w.r.t synthetic data sets.
    ------
    Input:
        group_pars: [X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars]
        star_params: (dict)
    Output:   
        lnprior (float)

    todo:
        incorporate errors into the prior
    """
    max_age = np.max(star_params['times'])
    av_error = 0.01

    for stdev in group_pars[6:10]:
        if stdev <= 0:
            return -np.inf
    for corr in group_pars[10:13]:
        if corr <= -1 or corr >= 1:
            return -np.inf

    age = group_pars[13]
    if age < 0 or age > max_age:
        return -np.inf
    
    return 0.0

def lnlike(group_pars, star_params, m_list=None):
    """
    Calculates the likelihood function of the data set given the proposed
    group model.
    ------
    Input:
        group_pars: [X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars]
        star_params: the traceback positions of the stars
        m_list: list of membership probabilites of stars to this group
    Output:
        lnlike (float)
    """
    lnlike = 0
    nstars = len(star_params['xyzuvw'])

    # charactersitic minimum for pos and vel dispersion
    min_p_disp = 1.0
    min_v_disp = 0.5

    # construct the required data from the group pars
    group_mn = group_pars[0:6]
    group_cov = generate_cov(group_pars)
    eig_vals = np.linalg.eigvalsh(group_cov)

    group_icov = np.linalg.inv(group_cov)
    group_icov_det = np.prod(np.linalg.eigvalsh(group_icov))

    # Ensure the icov matrix is singular and definite
    group_icov_eig_vals = np.linalg.eigvalsh(group_icov)
    if np.min(group_icov_eig_vals) < 0:
        print("Eig vals were negative")
        return -np.inf

    age = group_pars[-1]

    # Sanity check
    try:
        assert (
        abs(group_icov_det -1/np.prod(np.linalg.eigvalsh(group_cov)))
                /group_icov_det < 1e-5)
    except:
        pdb.set_trace()

    # Calculate eigenvalue priors
    lnlike += ln_eig_prior(min_p_disp, min_v_disp, eig_vals)

    # interpolate traceback data if required
    if group_pars[13] == 0:
        star_mns = star_params['xyzuvw'][:,0,:]
        star_icovs     = np.linalg.inv(star_params['xyzuvw_cov'][:,0,:,:])
        star_icov_dets = np.linalg.det(star_icovs)
    else:
        star_mns, star_covs =\
                              interp_cov(age, star_params)
        star_icovs = np.linalg.inv(star_covs)
        star_icov_dets = np.linalg.det(star_icovs)

    # calculate the overlap of each star with the given group
    overlaps = overlap.get_overlaps(group_icov, group_mn, group_icov_det,
                                    star_icovs, star_mns, star_icov_dets,
                                    nstars)

    if np.isnan(sum(overlaps)):
        #pdb.set_trace()
        return -np.inf
        #overlaps[np.where(np.isnan(overlaps))] = 0.0

    return np.sum(np.log(overlaps))

def lnprobfunc(group_pars, star_params):
    """
    The function called by the emcee which calculates how likely the proposed
    group model is given the data. This function ties together the prior
    and likelihood function.
    ------
    Input:
        group_pars: [X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars]
        star_params: the traceback positions of the stars
    Output:
        lnporb (float)
    """
    lp = lnprior(group_pars, star_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(group_pars, star_params)

def fit_one_group(star_params, nburnin, nsteps, init_pars=None,
                  m_list = None):
    """
    Use a monte carlo markov chain to establish the group parameters that
    maximise the likelihood function. Return the samples and lnprobs.
    ------
    Input:
        star_params: the traceback positions of the stars
        nburnin: number of burn in steps
        nsteps:  number of sampling steps
       [init_pars]: optional initial group params
       [m_list] : [nstars] array (float),
            the probabliity that each star belongs to this group
    Output:
        samples : [nsteps, nwalkers, npars] array
            the chain of samples
        lnprob : [nsteps, nwalkers] array
            the calculated lnprob of each sample
    """
    npar = 14
    nwalkers = 28   # hardcoded as twice the number of params in a group model
    
    # Initialise init_pars if needed
    # Note, the sdevs are in inverted form
    nstars = star_params['xyzuvw'].shape[0]
    if not init_pars:
        init_pars = [0,0,0,0,0,0,0.2,0.2,0.2,0.5,0,0,0,0]

    # Intialise membership_list if needed
    if not m_list:
        m_list = np.ones(nstars)

    # initialising the emcee sampler
    init_sdev = [1,1,1,1,1,1,
                 0.005, 0.005, 0.005, 0.005,
                 0.01,0.01,0.01,
                 1.0]
    pos = [init_pars+(np.random.random(size=len(init_sdev))- 0.5)*init_sdev
                                            for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(
                nwalkers, npar, lnprobfunc,
                args=[star_params])

    # Running the burn in, and saving poor chains by moving them to the best
    # location
    pos, lnprob, state = sampler.run_mcmc(pos, nburnin)
    best_chain = np.argmax(lnprob)
    poor_chains = np.where(lnprob < np.percentile(lnprob, 33))
    for ix in poor_chains:
        pos[ix] = pos[best_chain]
    sampler.reset()
    
    # running the sampling stage
    pos, final_lnprob, rstate = sampler.run_mcmc(pos, nsteps, rstate0=state)
    samples = sampler.chain
    lnprob = sampler.lnprobability
    # CONTINUE HERE
    
    return samples, lnprob

def run_fit(infile, nburnin, nsteps, ngroups=1):
    """
    Entry point for module. Given the traceback of stars, fit a set number
    of groups to the data. Left unfulfilled atm since only fitting a single
    group.
    """
    star_params = read_stars(infile)
    samples, lnprob = fit_one_group(star_params, nburnin, nsteps)
    return samples, lnprob
