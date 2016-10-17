"""This program takes an initial model for a stellar association and uses an affine invariant
Monte-Carlo to fit for the group parameters.

A group fitter, called after tracing orbits back.

This group fitter will find the best fit 6D error ellipse and best fit time for
the group formation based on Bayesian analysis, which in this case involves
computing overlap integrals. 
    
TODO:
0) Once the group is found, output the probability of each star being in the group.
1) Add in multiple groups 
2) Change from a group to a cluster, which can evaporate e.g. exponentially.
3) Add in a fixed background which is the Galaxy (from Robin et al 2003).

To use MPI, try:

mpirun -np 2 python fit_group.py

Note that this *doesn't* work yet due to a "pickling" problem.
"""

from __future__ import print_function, division

import emcee
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
try:
    import overlap #&TC
except:
    print("overlap not imported, SWIG not possible. Need to make in directory...")
import time    #&TC
from emcee.utils import MPIPool
   
def compute_overlap(A,a,A_det,B,b,B_det):
    """Compute the overlap integral between a star and group mean + covariance matrix
    in six dimensions, including some temporary variables for speed and to match the 
    notes.
    
    This is the first function to be converted to a C program in order to speed up."""

    #Preliminaries - add matrices together. This might make code more readable? 
    #Or might not.
    ApB = A + B
    AapBb = np.dot(A,a) + np.dot(B,b)
    
    #Compute determinants.
    ApB_det = np.linalg.det(ApB)
    
    #Error checking (not needed in C once shown to work?) This shouldn't ever happen, as 
    #the determinants of the sum of positive definite matrices is
    #greater than the sum of their determinants    
    if (ApB_det < 0) | (B_det<0):
        pdb.set_trace()
        return -np.inf
    
    #Solve for c
    c = np.linalg.solve(ApB, AapBb)
    
    #Compute the overlap formula.
    overlap = np.exp(-0.5*(np.dot(b-c,np.dot(B,b-c)) + \
                           np.dot(a-c,np.dot(A,a-c)) )) 
    overlap *= np.sqrt(B_det*A_det/ApB_det)/(2*np.pi)**3.0
    
    return overlap
   
def read_stars(infile):
    """Read stars from a previous pickle file into a dictionary.
    
    The input is an error ellipse in 6D (X,Y,Z,U,V,W) of a list of stars, at
    a bunch of times in the past.
    
    Parameters
    ----------
    infile: string
        input pickle file
        
    Returns
    -------
    star_dictionary: dict
        stars: (nstars) high astropy table including columns as documented in the Traceback class.
        times: (ntimes) numpy array, containing times that have been traced back, in Myr
        xyzuvw (nstars,ntimes,6) numpy array, XYZ in pc and UVW in km/s
        xyzuvw_cov (nstars,ntimes,6,6) numpy array, covariance of xyzuvw
    """
    if len(infile)==0:
        print("Input a filename...")
        raise UserWarning
    
    #Stars is an astropy.Table of stars
    fp = open(infile,'r')
    (stars,times,xyzuvw,xyzuvw_cov)=pickle.load(fp)
    fp.close()

    #Preliminaries. 
    #Create the inverse covariances and other globals.
    ns = len(stars)    #Number of stars
    nt = len(times)    #Number of times.
    xyzuvw_icov = np.empty( (ns,nt,6,6) )
    xyzuvw_icov_det = np.empty( (ns,nt) )
    #Fill up the inverse covariance matrices.
    for i in range(ns):
        for j in range(nt):
            xyzuvw_icov[i,j]     = np.linalg.inv(xyzuvw_cov[i,j])
            xyzuvw_icov_det[i,j] = np.linalg.det(xyzuvw_icov[i,j])
    return dict(stars=stars,times=times,xyzuvw=xyzuvw,xyzuvw_cov=xyzuvw_cov,xyzuvw_icov=xyzuvw_icov,xyzuvw_icov_det=xyzuvw_icov_det)

   
def lnprob_one_group(x, star_params, background_density=2e-12,use_swig=True,t_ix = 0,return_overlaps=False,\
    return_cov=False, min_axis=2.0,min_v_disp=0.5,debug=False):
    """Compute the log-likelihood for a fit to a group.

    The x variables are:
    xyzuvw (6), then xyz standard deviations (3), uvw_symmetrical_std (1), xyz_correlations (3)

    A 14th variable, if present, is the time at which the calculation is made. If not given, the
    calculation is made at a fixed time index t_ix.

    The probability of a model is the product of the probabilities
    overlaps of every star in the group. 

    Parameters
    ----------
    x : array-like
        The group parameters, which are...
        x[0] to x[5] : xyzuvw
        x[6] to x[8] : positional variances in x,y,z
        x[9]  : velocity dispersion (symmetrical for u,v,w)
        x[10] to x[12] :  correlations between x,y,z
        x[13] : (optional) birth time of group in Myr. 

    background_density :
        The density of a background stellar population, in
        units of pc**(-3)*(km/s)**(-3). 
    
    t_ix : int
        Time index (in the past) where we are computing the probabilities.
    
    return_overlaps : bool  
        Return the overlaps (rather than the log probability)
    
    return_cov : bool
        Return the covariance (rather than the log probability)
    
    min_axis : float
        Minimum allowable position dispersion for the cluster in parsecs
    
    min_v_disp : float
        Minimum allowable cluster velocity dispersion in km/s.
    """
    practically_infinity = np.inf#1e20
    
    xyzuvw = star_params['xyzuvw']
    xyzuvw_cov = star_params['xyzuvw_cov']
    xyzuvw_icov = star_params['xyzuvw_icov']
    xyzuvw_icov_det = star_params['xyzuvw_icov_det']
    times = star_params['times']
    ns = len(star_params['stars'])    #Number of stars
    nt = len(times)    #Number of times.

    #See if we have a time in Myr in the input vector, in which case we have
    #to interpolate in time. Otherwise, just choose a single time snapshot given 
    #by the input index t_ix.
    if len(x)>13:
        #If the input time is outside our range of traceback times, return
        #zero likelihood.
        if ( (x[13] < min(times)) | (x[13] > max(times))):
            return -np.inf 
        #Linearly interpolate in time to get bs and Bs
        #Note that there is a fast scipy package (in ndimage?) that is good for this.
        ix = np.interp(x[13],times,np.arange(nt))
        ix0 = np.int(ix)
        frac = ix-ix0
        bs     = xyzuvw[:,ix0]*(1-frac) + xyzuvw[:,ix0+1]*frac
        cov    = xyzuvw_cov[:,ix0]*(1-frac) + xyzuvw_cov[:,ix0+1]*frac
        Bs     = np.linalg.inv(cov)
        B_dets = np.linalg.det(Bs)
    else:
        #Extract the time that we really care about.
        #The result is a (ns,6) array for bs, and (ns,6,6) array for Bs.
        bs     = xyzuvw[:,t_ix]
        Bs     = xyzuvw_icov[:,t_ix]
        B_dets = xyzuvw_icov_det[:,t_ix]

    #Sanity check inputs for out of bounds. If so, return zero likelihood.
    if (np.min(x[6:9])<=min_axis):
        if debug:
            print("Positional Variance Too Low...")
        return -practically_infinity
    if (np.min(x[9])<min_v_disp):
        if debug:
            print("Velocity Variance Too Low...")
        return -practically_infinity
    if (np.max(np.abs(x[10:13])) >= 1):
        if debug:
            print("Correlations above 1...")
        return -practically_infinity       

    #Create the group_mn and group_cov from x. This looks a little tricky 
    #because we're inputting correlations rather than elements of the covariance
    #matrix.
    #https://en.wikipedia.org/wiki/Correlation_and_dependence
    x = np.array(x)
    group_mn = x[0:6]
    group_cov = np.eye( 6 )
    #Fill in correlations
    group_cov[np.tril_indices(3,-1)] = x[10:13]
    group_cov[np.triu_indices(3,1)] = x[10:13]
    #Convert correlation to covariance for position.
    for i in range(3):
        group_cov[i,:3] *= x[6:9]
        group_cov[:3,i] *= x[6:9]
    #Convert correlation to covariance for velocity.
    for i in range(3,6):
        group_cov[i,3:] *= x[9]
        group_cov[3:,i] *= x[9]

    #Allow this covariance matrix to be returned.
    if return_cov:
        return group_cov

    #Enforce some sanity check limits on prior...
    if (np.min(np.linalg.eigvalsh(group_cov[:3,:3])) < min_axis**2):
        if debug:
            print("Minimum positional covariance too small in one direction...")
        return -practically_infinity

    #Invert the group covariance matrix and check for negative eigenvalues
    group_icov = np.linalg.inv(group_cov)
    group_icov_eig = np.linalg.eigvalsh(group_icov)
    if np.min(group_icov_eig) < 0:
        if debug:
            print("Numerical error in inverse covariance matrix!")
        return -practically_infinity
    group_icov_det = np.prod(group_icov_eig)

    #Before starting, lets set the prior probability
    #Given the way we're sampling the covariance matrix, I'm
    #really not sure this is correct! But it is pretty close...
    #it looks almost like 1/(product of standard deviations).
    #See YangBerger1998
    lnprob=np.log(np.abs(group_icov_det)**3.5)

    #overlaps_start = time.clock()
    #Now loop through stars, and save the overlap integral for every star.
    overlaps = np.empty(ns)
    if use_swig:
        if (False):
            #NOT TESTED, JUST TO GIVE MIKE AN IDEA OF THE LAYOUT OF THE FUNCTION CALL
            overlaps = overlap.get_overlap(group_icov, group_mn, group_icov_det,
                                            Bs, bs, B_dets, ns)
            #note 'ns' at end, see 'overlap.c' for documentation
            lnprob = lnprob + np.sum(np.log(background_density + overlaps))
        else:
            for i in range(ns):
                overlaps[i] = overlap.get_overlap(group_icov,
                                                  group_mn,
                                                  group_icov_det,
                                                  Bs[i],
                                                  bs[i],
                                                  B_dets[i]) #&TC
                lnprob += np.log(background_density + overlaps[i])
    else:
        for i in range(ns):
            overlaps[i] = compute_overlap(group_icov,group_mn,group_icov_det,Bs[i],bs[i],B_dets[i])
            lnprob += np.log(background_density + overlaps[i])
    
    #print (time.clock() - overlaps_start)

    if return_overlaps:
        return overlaps    
    
    return lnprob

def lnprob_one_cluster(x, star_params, use_swig=False, return_overlaps=False, \
    min_axis=2.0, min_v_disp=0.5, debug=False):
    """Compute the log-likelihood for a fit to a cluster. A cluster is defined as a group that decays 
    exponentially in time.

    The minimal set of x variables are:
    xyzuvw (6), the core radius (1),  
    the tidal radius now (1) [starts at 1.5 times the core radius], the initial velocity 
    dispersion (1) [decays according to density ** 0.5], 
    the birth time (1), the central density decay time (1),
    
    The probability of a model is the product of the probabilities
    overlaps of every star in the group. 

    Parameters
    ----------
    x : array-like
        The group parameters, which are...
        x[0] to x[5] : xyzuvw at the CURRENT time.
        x[6]  : Core radius (constant with time)
        x[7]  : Tidal radius at current epoch.
        x[8]  : Initial velocity dispersion
        x[9]  : Birth time
        x[10] : Central density 1/e decay time.
        x[11] : Initial central density [for now as a multiplier of the 
                background density in units of pc^{-3} km^{-3} s^3

    star_params : astropy table

    return_overlaps : bool  
        Return the overlaps (rather than the log probability)
    
    return_cov : bool
        Return the covariance (rather than the log probability)
    
    min_axis : float
        Minimum allowable position dispersion for the cluster in parsecs
    
    min_v_disp : float
        Minimum allowable cluster velocity dispersion in km/s.
    """
    practically_infinity = 1e20
    
    #Extract the key parameters in shorthand from star_params.
    xyzuvw = star_params['xyzuvw']
    xyzuvw_cov = star_params['xyzuvw_cov']
    xyzuvw_icov = star_params['xyzuvw_icov']
    xyzuvw_icov_det = star_params['xyzuvw_icov_det']
    times = star_params['times']
    ns = len(star_params['stars'])    #Number of stars
    nt = len(times)    #Number of times.

    #Sanity check inputs for out of bounds...
    if (np.min(x[6:8])<=min_axis):
        if debug:
            print("Positional Variance Too Low...")
        return -practically_infinity
    if (x[8]<min_v_disp):
        if debug:
            print("Velocity Variance Too Low...")
        return -practically_infinity

    #Trace the cluster backwards forwards in time. For every timestep, we have value of 
    #xyzuvw for the cluster. Covariances are simply formed from the radius and dispersion - 
    #they are symmetrical
    
    #!!! THIS SHOULD USE THE TRACEBACK MODULE, AND IS A JOB FOR JONAH TO TRY !!!
    
    #Now loop through stars, and save the overlap integral for every star.
    overlaps = np.empty(ns)
    for i in range(ns):
        #!!! Check if the spatial overlap is significant. If it is, find the time of 
        #overlap and the parameters of the cluster treated as two groups at this time. !!!
        spatial_overlap_is_significant = False
        if spatial_overlap:
            #!!! the "group" parameters below need to be set !!!
            if use_swig:
                overlaps[i] = overlap.get_overlap(group_icov.flatten().tolist(),
                                              group_mn.flatten().tolist(),
                                              group_icov_det,
                                              Bs[i].flatten().tolist(),
                                              bs[i].flatten().tolist(),
                                              B_dets[i]) 
            else:
                overlaps[i] = compute_overlap(group_icov,group_mn,group_icov_det,Bs[i],bs[i],B_dets[i])
        lnprob += np.log(1 + overlaps[i]*x[11])

    if return_overlaps:
        return overlaps    
    
    return lnprob

def fit_one_group(star_params, init_mod=np.array([ -6.574, 66.560, 23.436, -1.327,-11.427, -6.527, \
    10.045, 10.319, 12.334,  0.762,  0.932,  0.735,  0.846, 20.589]),\
        nwalkers=100,nchain=1000,nburn=200, return_sampler=False,pool=None,\
        init_sdev = np.array([1,1,1,1,1,1,1,1,1,.01,.01,.01,.1,1]), background_density=2e-12, use_swig=True, \
        plotit=False):
    """Fit a single group, using a affine invariant Monte-Carlo Markov chain.
    
    Parameters
    ----------
    star_params: dict
        A dictionary of star parameters from read_stars. This should of course be a
        class, but it doesn't work with MPI etc as class instances are not 
        "pickleable"
        
    init_mod : array-like
        See lnprob_one_group for parameter definitions.
        
    nwalkers : int
        Number of walkers to characterise the parameter covariance matrix. Has to be
        at least 2 times the number of dimensions.
    
    nchain : int
        Number of elements in the chain. For characteristing a distribution near a 
        minimum, 1000 is a rough minimum number (giving ~10% uncertainties on 
        standard deviation estimates).
        
    nburn : int
        Number of burn in steps, before saving any chain output. If the beam acceptance
        fraction is too low (e.g. significantly lower in burn in than normal, e.g. 
        less than 0.1) then this has to be increased.
    
    Returns
    -------
    best_params: array-like
        The best set of group parameters.
    sampler: emcee.EmsembleSampler
        Returned if return_sampler=True
    """
    nparams = len(init_mod)
    #Set up the MCMC...
    ndim=nparams

    #Set an initial series of models
    p0 = [init_mod + (np.random.random(size=ndim) - 0.5)*init_sdev for i in range(nwalkers)]

    #NB we can't set e.g. "threads=4" because the function isn't "pickleable"
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_one_group,pool=pool,args=[star_params,background_density,use_swig])

    #Burn in...
    pos, prob, state = sampler.run_mcmc(p0, nburn)
    print("Mean burn-in acceptance fraction: {0:.3f}"
                    .format(np.mean(sampler.acceptance_fraction)))

    sampler.reset()

    #Run...
    sampler.run_mcmc(pos, nchain)
    if plotit:
        plt.figure(1)
        plt.clf()
        plt.plot(sampler.lnprobability.T)

    #Best Model
    best_ix = np.argmax(sampler.flatlnprobability)
    print('[' + ",".join(["{0:7.3f}".format(f) for f in sampler.flatchain[best_ix]]) + ']')
    overlaps = lnprob_one_group(sampler.flatchain[best_ix], star_params,return_overlaps=True,use_swig=use_swig)
    group_cov = lnprob_one_group(sampler.flatchain[best_ix], star_params,return_cov=True,use_swig=use_swig)
    np.sqrt(np.linalg.eigvalsh(group_cov[:3,:3]))
    ww = np.where(overlaps < background_density)[0]
    print("The following stars have very small overlaps with the group...")
    print(star_params['stars'][ww]['Name'])

    print("Mean acceptance fraction: {0:.3f}"
                    .format(np.mean(sampler.acceptance_fraction)))

    if plotit:
        plt.figure(2)       
        plt.clf()         
        plt.hist(sampler.chain[:,:,-1].flatten(),20)
    
    #pdb.set_trace()
    
    return sampler.flatchain[best_ix]
        
#Some test calculations applicable to the ARC DP17 proposal.
if __name__ == "__main__":
    star_params = read_stars("traceback_save.pkl")
    
    using_mpi = True
    try:
        # Initialize the MPI-based pool used for parallelization.
        pool = MPIPool()
    except:
        print("MPI doesn't seem to be installed... maybe install it?")
        using_mpi = False
        pool=None
    
    if using_mpi:
        if not pool.is_master():
            # Wait for instructions from the master process.
            pool.wait()
            sys.exit(0)
        else:
            print("MPI available! - call this with e.g. mpirun -np 4 python fit_group.py")
    
    beta_pic_group = np.array([-6.574, 66.560, 23.436, -1.327,-11.427, -6.527,\
        10.045, 10.319, 12.334,  0.762,  0.932,  0.735,  0.846, 20.589])
    plei_group = np.array([116.0,27.6, -27.6, 4.7, -23.1, -13.2, 20, 20, 20,\
                        3, 0, 0, 0, 70])

    dummy = lnprob_one_group(beta_pic_group, star_params, use_swig=True)
#    dummy = lnprob_one_group(plei_group, star_params, background_density=1e-10, use_swig=False)
        
    fitted_params = fit_one_group(star_params, pool=pool, use_swig=True)
    
    if using_mpi:
        # Close the processes.
        pool.close()
