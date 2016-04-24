"""This program takes an initial model for a stellar association and uses an affine invariant
Monte-Carlo to fit for the group parameters.

To use MPI, try:

mpirun -np 2 python examples/mpi.py

Note that this *doesn't* work yet due to a "pickling" problem.
"""

from __future__ import print_function, division

import emcee
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
from emcee.utils import MPIPool
plt.ion()
   
def compute_overlap(A,a,B,b,A_det):
    """Compute the overlap integral between a star and group mean + covariance matrix
    in six dimensions, including some temporary variables for speed and to match the 
    notes.
    
    This is the first function to be converted to a C program in order to speed up."""
    #Preliminaries - add matrices together. This might make code more readable? 
    #Or might not.
    ApB = A + B
    AapBb = np.dot(A,a) + np.dot(B,b)
    
    #Compute determinants.
    B_det   = np.linalg.det(B)
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
   
class GroupFitter():
    """A group fitter, called after tracing orbits back.
    
    The input is an error ellipse in 6D (X,Y,Z,U,V,W) of a list of stars, at
    a bunch of times in the past.
    
    This group fitter will find the best fit 6D error ellipse and best fit time for
    the group formation based on Bayesian analysis, which in this case involves
    computing overlap integrals. 
    
    TODO:
    1) Add in multiple groups 
    2) Change from a group to a cluster, which can evaporate e.g. exponentially.
    
    Parameters
    ----------
    infile: string
        input pickle file
    """
    def __init__(self,infile=""):
        if len(infile)==0:
            print("Direct variable input not implemented! input a file...")
            raise UserWarning
        
        #Stars is an astropy.Table of stars
        fp = open(infile,'r')
        (self.stars,self.times,self.xyzuvw,self.xyzuvw_cov)=pickle.load(fp)
        fp.close()

        #Preliminaries. 
        #Create the inverse covariances and other globals.
        self.ns = len(self.stars)    #Number of stars
        self.nt = len(self.times)    #Number of times.
        self.xyzuvw_icov = np.empty( (self.ns,self.nt,6,6) )
        #Fill up the inverse covariance matrices.
        for i in range(self.ns):
            for j in range(self.nt):
                self.xyzuvw_icov[i,j] = np.linalg.inv(self.xyzuvw_cov[i,j])

    def lnprob_one_group(self, x, background_density=2e-12,t_ix = 20,return_overlaps=False,\
        return_cov=False, min_axis=2.0,min_v_disp=0.5):
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
        xyzuvw = self.xyzuvw
        xyzuvw_cov = self.xyzuvw_cov
        xyzuvw_icov = self.xyzuvw_icov
    
        #See if we have to interpolate in time.
        if len(x)>13:
            if ( (x[13] < 0) | (x[13] >= self.nt-1)):
                return -np.inf
            #Linearly interpolate in time to get bs and Bs
            #Note that there is a fast scipy package (in ndimage?) that is good for this.
            ix = np.interp(x[13],self.times,np.arange(self.nt))
            ix0 = np.int(ix)
            frac = ix-ix0
            bs     = xyzuvw[:,ix0]*(1-frac) + xyzuvw[:,ix0+1]*frac
            cov    = xyzuvw_cov[:,ix0]*(1-frac) + xyzuvw_cov[:,ix0+1]*frac
            Bs     = np.linalg.inv(cov)
        else:
            #Extract the time that we really care about.
            #The result is a (self.ns,6) array for bs, and (ns,6,6) array for Bs.
            bs     = xyzuvw[:,t_ix]
            Bs     = xyzuvw_icov[:,t_ix]

        #Sanity check inputs for out of bounds...
        if (np.min(x[6:9])<=min_axis):
            return -np.inf
        if (np.min(x[9])<min_v_disp):
            return -np.inf
        if (np.max(np.abs(x[10:13])) >= 1):
            return -np.inf       
    
        #Create the group_mn and group_cov from x.
        x = np.array(x)
        group_mn = x[0:6]
        group_cov = np.eye( 6 )
        group_cov[np.tril_indices(3,-1)] = x[10:13]
        group_cov[np.triu_indices(3,1)] = x[10:13]
        for i in range(3):
            group_cov[i,:3] *= x[6:9]
            group_cov[:3,i] *= x[6:9]
        for i in range(3,6):
            group_cov[i,3:] *= x[9]
            group_cov[3:,i] *= x[9]

        #Allow this covariance matrix to be returned.
        if return_cov:
            return group_cov

        #Enforce some sanity check limits on prior...
        if (np.min(np.linalg.eigvalsh(group_cov[:3,:3])) < min_axis**2):
            return -np.inf

        #Invert the group covariance matrix and check for negative eigenvalues
        group_icov = np.linalg.inv(group_cov)
        group_icov_eig = np.linalg.eigvalsh(group_icov)
        if np.min(group_icov_eig) < 0:
            return -np.inf
        group_icov_det = np.prod(group_icov_eig)
    
        #Before starting, lets set the prior probability
        #Given the way we're sampling the covariance matrix, I'm
        #really not sure this is correct! But it is pretty close...
        #it looks almost like 1/(product of standard deviations).
        #See YangBerger1998
        lnprob=np.log(np.abs(group_icov_det)**3.5)

        #Now loop through stars, and save the overlap integral for every star.
        overlaps = np.empty(self.ns)
        for i in range(self.ns):
            overlaps[i] = compute_overlap(group_icov,group_mn,Bs[i],bs[i],group_icov_det)
            lnprob += np.log(background_density + overlaps[i])
        if return_overlaps:
            return overlaps    

        return lnprob

    def fit_one_group(self,init_mod=np.array([ -6.574, 66.560, 23.436, -1.327,-11.427, -6.527, \
        10.045, 10.319, 12.334,  0.762,  0.932,  0.735,  0.846, 20.589]),\
            nwalkers=100,nchain=1000,nburn=200, return_sampler=False,pool=None):
        """Fit a single group, using a affine invariant Monte-Carlo Markov chain.
        
        Parameters
        ----------
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
        p0 = [init_mod + np.random.random(size=ndim) - 0.5 for i in range(nwalkers)]

        #NB we can't set e.g. "threads=4" because the function isn't "pickleable"
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob_one_group,pool=pool)

        #Burn in...
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        print("Mean burn-in acceptance fraction: {0:.3f}"
                        .format(np.mean(sampler.acceptance_fraction)))
        sampler.reset()

        #Run...
        sampler.run_mcmc(pos, nchain)
        plt.figure(1)
        plt.clf()
        plt.plot(sampler.lnprobability.T)

        #Best Model
        best_ix = np.argmax(sampler.flatlnprobability)
        print('[' + ",".join(["{0:7.3f}".format(f) for f in sampler.flatchain[best_ix]]) + ']')
        overlaps = self.lnprob_one_group(sampler.flatchain[best_ix],return_overlaps=True)
        group_cov = self.lnprob_one_group(sampler.flatchain[best_ix],return_cov=True)
        np.sqrt(np.linalg.eigvalsh(group_cov[:3,:3]))
        ww = np.where(overlaps < 2e-12)[0]
        print("The following stars have very small overlaps with the group...")
        print(self.stars[ww]['Name'])

        print("Mean acceptance fraction: {0:.3f}"
                        .format(np.mean(sampler.acceptance_fraction)))

        plt.figure(2)       
        plt.clf()         
        plt.hist(sampler.chain[:,:,-1].flatten(),20)
        
        return sampler.flatchain[best_ix]
        
#Some test calculations applicable to the ARC DP17 proposal.
if __name__ == "__main__":
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
        
    fit_betpic = GroupFitter("traceback_save.pkl")
    fitted_params = fit_betpic.fit_one_group(pool=pool)
    
    if using_mpi:
        # Close the processes.
        pool.close()