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
    import astropy.io.fits as pyfits
except:
    import pyfits

try:
    import _overlap as overlap #&TC
except:
    print("overlap not imported, SWIG not possible. Need to make in directory...")
import time    #&TC
from emcee.utils import MPIPool

try:                # don't know why we use xrange to initialise walkers
    xrange
except NameError:
    xrange = range

class MVGaussian(object):
    """
        This class aims to encapsulate the complicated maths used to convert
        stellar measurements and group parameters into a 6D multivariate
        gaussian. It may eventually have other cool features.
        params is used to refer to the following:
            x[0] to x[5] : xyzuvw
            x[6] to x[8] : positional variances in x,y,z
            x[9]  : velocity dispersion (symmetrical for u,v,w)
            x[10] to x[12] :  correlations between x,y,z
    """
    params   = None   #UVWXYZ etc parameters
    mean     = None   # mean of the MVGausssian
    icov     = None   # inverse covariance matrix
    icov_det = None
    
    def __init__(self, params): 
        self.params = np.array(params)
        self.generateIcovAndMean()

    def generateIcovAndMean(self):
        self.mean = self.params[0:6]        

        cov = np.eye( 6 )
        #Fill in correlations
        cov[np.tril_indices(3,-1)] = self.params[10:13]
        cov[np.triu_indices(3,1)] = self.params[10:13]
        #Convert correlation to covariance for position.
        for i in range(3):
            cov[i,:3] *= self.params[6:9]
            cov[:3,i] *= self.params[6:9]
        #Convert correlation to covariance for velocity.
        for i in range(3,6):
            cov[i,3:] *= self.params[9]
            cov[3:,i] *= self.params[9]
        #Generate inverse cov matrix and its determinant
        self.icov = np.linalg.inv(cov)
        self.icov_det = np.prod(np.linalg.eigvalsh(self.icov))

    def __str__(self):
        return "MVGauss with icov:\n{}\nand icov_det: {}".format(
                    self.icov, self.icov_det)

class Star(MVGaussian):
    """
        Specific to stars and interpolation nonsense
    """

class Group(MVGaussian):
    """
        Encapsulates the various forms a group model can take
        for example it may be one with fixed parameters and just amplitude
        varying.
    """
    amplitude = None
    age       = None

    def __init__(self, params, amplitude, age): 
        super(self.__class__,self).__init__(params)
        self.amplitude = amplitude
        self.age = age

class GroupFitter:
    """
        This class will find the best fitting group models to a set of stars.
        Group models are 6-dimensional multivariate Gausssians which are
        designed to find the instance in time when a set of stars occupied
        the smallest volume
    """
    # Data variables
    NDIM    = 6       # number of dimensions for each 'measured' star
    NGROUPS = None       # number of groups in the data
    GROUPS       = []
    FIXED_GROUPS = []
    NSTARS      = None
    STAR_PARAMS = None
    STAR_MNS    = None   # a [NSTARSxNTIMESx6] matrix
    STAR_ICOVS  = None   # a [NSTARSxNTIMESx6x6] matrix
    STAR_ICOV_DETS = None  # a [NSTARS*NTIMES] matrix

    # emcee parameters
    burnin = 20
    steps  = 50
    NWALKERS = 30
    NPAR = 13

    # Fitting variables
    samples  = None
    means    = None  # modelled means [a NGROUPSx6 matrix]
    cov_mats = None # modelled cov_matrices [a NGROUPSx6x6 matrix]
    weights  = None # the amplitude of each gaussian [a NGROUP matrix]
    best_fit = np.zeros(14) # best fitting group parameters, same order as 'pars'
    
    def __init__(self, ngroups=1, infile='results/bp_TGAS2_traceback_save.pkl'):
        self.NGROUPS = ngroups
        self.STAR_PARAMS = self.read_stars(infile)
        pdb.set_trace()
        self.NSTARS = len(self.STAR_PARAMS['xyzuvw'])
        init_group_params = [-15.41, -17.22, -21.32, -4.27, -14.39, -5.83,
                              73.34, 51.61, 48.83,
                              7.20,
                             -0.21, -0.09, 0.12]
        self.GROUPS = [None] * self.NGROUPS
        self.GROUPS[0] = Group(init_group_params, 1.0, 0.0)

    def __str__(self):
        return "A groupfitter with {} stars".format(self.NSTARS)

    def read_stars(self, infile):
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
        #Create the inverse covariances to save time.
        xyzuvw_icov = np.linalg.inv(xyzuvw_cov)
        xyzuvw_icov_det = np.linalg.det(xyzuvw_icov)

        #Store key data in globals
        self.STAR_MNS       = xyzuvw
        self.STAR_ICOVS     = xyzuvw_icov
        self.STAR_ICOV_DETS = xyzuvw_icov_det 
    
        return dict(stars=stars,times=times,xyzuvw=xyzuvw,xyzuvw_cov=xyzuvw_cov,
                       xyzuvw_icov=xyzuvw_icov,xyzuvw_icov_det=xyzuvw_icov_det)

    def lnprior(self, pars):
        return 0.0

    def lnlike(self, pars):
        model_group = Group(pars, 1.0, 0.0)
        group_icov = model_group.icov
        group_mn   = model_group.mean
        group_icov_det = model_group.icov_det

        #extract the time we're interested in
        star_icovs = self.STAR_ICOVS[:,0,:,:]
        star_mns = self.STAR_MNS[:,0,:]
        star_icov_dets = self.STAR_ICOV_DETS[:,0]

        # use swig module to calculate overlaps of each star with model
        overlaps = overlap.get_overlaps(group_icov, group_mn, group_icov_det,
                                        star_icovs, star_mns,
                                        star_icov_dets, self.NSTARS)

        # calculate the product of the likelihoods
        return np.sum(np.log(overlaps))
    
    def lnprob(self, pars):
        """Compute the log-likelihood for a fit to a group.
           pars are the parameters being fitted for by MCMC 

            for simplicity, simply looking at time=0
        """
        lp = self.lnprior(pars)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(pars)

    def fitGroups(self):
        init_mod = self.GROUPS[0].params
        init_sdev = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.01, 0.01, 0.01]
        p0 = [init_mod + (np.random.random(size=self.NPAR) - 0.5)*init_sdev
                                                    for i in range(self.NWALKERS)]

        self.sampler = emcee.EnsembleSampler(self.NWALKERS, self.NPAR, self.lnprob)

        pos, lnprob, state = self.sampler.run_mcmc(p0, self.burnin)

        best_chain = np.argmax(lnprob)
        poor_chains = np.where(lnprob < np.percentile(lnprob, 33))
        for ix in poor_chains:
            pos[ix] = pos[best_chain]
    
        self.sampler.reset()
        self.sampler.run_mcmc(pos, self.steps, rstate0=state)

        #Best Model
        best_ix = np.argmax(self.sampler.flatlnprobability)
        print('[' + ",".join(["{0:7.3f}".format(f) for f in self.sampler.flatchain[best_ix]]) + ']')
        #overlaps = lnprob_one_group(sampler.flatchain[best_ix], star_params,return_overlaps=True,use_swig=use_swig)

    def interp_icov(self, target_time):
        """
        Interpolate in time to get the xyzuvw vector and incovariance matrix.
        """
        times = self.STAR_PARAMS['times']
        ix = np.interp(target_time, times, np.arange(len(times)))
        ix0 = np.int(ix)
        frac = ix-ix0
        interp_mns       = self.STAR_MNS[:,ix0]*(1-frac) +\
                                self.STAR_MNS[:,ix0+1]*frac
        interp_icovs     = self.STAR_ICOVS[:,ix0]*(1-frac) +\
                                self.STAR_ICOVS[:,ix0+1]*frac
        interp_icov_dets = self.STAR_ICOV_DETS[:,ix0]*(1-frac) +\
                                self.STAR_ICOV_DETS[:,ix0+1]*frac
        return interp_mns, interp_icovs, interp_icov_dets
       

def interp_cov(target_time, star_params):
    """
    Interpolate in time to get a xyzuvw vector and covariance matrix.
    
    Note that there is a fast scipy package (in ndimage?) that might be good for this.
    """         
    times = star_params['times']
    ix = np.interp(target_time,times,np.arange(len(times)))
    ix0 = np.int(ix)
    frac = ix-ix0
    bs     = star_params['xyzuvw'][:,ix0]*(1-frac) + star_params['xyzuvw'][:,ix0+1]*frac
    cov    = star_params['xyzuvw_cov'][:,ix0]*(1-frac) + star_params['xyzuvw_cov'][:,ix0+1]*frac
    return bs, cov


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
        Initial mean of models used to fit the group. See lnprob_one_group for parameter definitions.

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
        plt.savefig("plots/lnprobability.eps")
        plt.pause(0.001)

    #Best Model
    best_ix = np.argmax(sampler.flatlnprobability)
    print('[' + ",".join(["{0:7.3f}".format(f) for f in sampler.flatchain[best_ix]]) + ']')
    overlaps = lnprob_one_group(sampler.flatchain[best_ix], star_params,return_overlaps=True,use_swig=use_swig)
    group_cov = lnprob_one_group(sampler.flatchain[best_ix], star_params,return_cov=True,use_swig=use_swig)
    np.sqrt(np.linalg.eigvalsh(group_cov[:3,:3]))
    ww = np.where(overlaps < background_density)[0]
    print("The following {0:d} stars are more likely not group members...".format(len(ww)))
    try:
        print(star_params['stars'][ww]['Name'])
    except:
       print(star_params['stars'][ww]['Name1'])

    print("Mean acceptance fraction: {0:.3f}"
                    .format(np.mean(sampler.acceptance_fraction)))

    if plotit:
        plt.figure(2)       
        plt.clf()         
        plt.hist(sampler.chain[:,:,-1].flatten(),20)
        plt.savefig("plots/distribution_of_ages.eps")
    
    #pdb.set_trace()
    if return_sampler:
        return sampler
    else:
        return sampler.flatchain[best_ix]


    #Set an initial series of models
    p0 = [init_mod + (np.random.random(size=ndim) - 0.5)*init_sdev for i in range(nwalkers)]

    #NB we can't set e.g. "threads=4" because the function isn't "pickleable"
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_two_groups,pool=pool,args=[star_params,use_swig])

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
        plt.savefig("plots/lnprobability.eps")
        plt.pause(0.001)

    #Best Model
    best_ix = np.argmax(sampler.flatlnprobability)
    print('[' + ",".join(["{0:7.3f}".format(f) for f in sampler.flatchain[best_ix]]) + ']')
#    overlaps = lnprob_one_group(sampler.flatchain[best_ix], star_params,return_overlaps=True,use_swig=use_swig)
#    group_cov = lnprob_one_group(sampler.flatchain[best_ix], star_params,return_cov=True,use_swig=use_swig)
#    np.sqrt(np.linalg.eigvalsh(group_cov[:3,:3]))
#    ww = np.where(overlaps < background_density)[0]
#    print("The following {0:d} stars are more likely not group members...".format(len(ww)))
#    try:
#        print(star_params['stars'][ww]['Name'])
#    except:
#       print(star_params['stars'][ww]['Name1'])
#
    print("Mean acceptance fraction: {0:.3f}"
                    .format(np.mean(sampler.acceptance_fraction)))

    if plotit:
        plt.figure(2)       
        plt.clf()         
        plt.hist(sampler.chain[:,:,-1].flatten(),20)
        plt.savefig("plots/distribution_of_ages.eps")
    
    #pdb.set_trace()
    if return_sampler:
        return sampler
    else:
        return sampler.flatchain[best_ix]

    if return_overlaps:
        return (g1_overlaps, g2_overlaps, bg_overlaps)
    
    return lnprob
