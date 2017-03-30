"""This program takes an initial model for a stellar association and uses an affine invariant
Monte-Carlo to fit for the group parameters.

A group fitter, called after tracing orbits back.

This group fitter will find the best fit 6D error ellipse and best fit time for
the group formation based on Bayesian analysis, which in this case involves
computing overlap integrals. 
    
TODO:
0) Save a log, save the samples
1) Make corner plot generic
2) make input parameters scale invariant
    - use arccos/arcsin for correlations e.g., 1/x for pos/vel dispersion
    - then tidy up samples at end by reconverting into "physical" parameters
3) Work out actual physical constraints for correlations

To use MPI, try:

mpirun -np 2 python fit_group.py

Note that this *doesn't* work yet due to a "pickling" problem.

Must be procedural for MPI to work
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
        
        for stdev in self.params[6:10]:
            try:
                assert(stdev > 0.0), "negative stdev"
            except:
                print("Negative stdev")
                pdb.set_trace()

        cov = np.eye( 6 )
        #Fill in correlations
        cov[np.tril_indices(3,-1)] = self.params[10:13]
        cov[np.triu_indices(3,1)] = self.params[10:13]
        #Note that 'pars' stores the inverse of the standard deviation
        #Convert correlation to covariance for position.
        for i in range(3):
            cov[i,:3] *= 1 / self.params[6:9]
            cov[:3,i] *= 1 / self.params[6:9]
        #Convert correlation to covariance for velocity.
        for i in range(3,6):
            cov[i,3:] *= 1 / self.params[9]
            cov[3:,i] *= 1 / self.params[9]
        #Generate inverse cov matrix and its determinant

        neg_cov = 0 + cov
        neg_cov[np.tril_indices(3,-1)] *= -1
        neg_cov[np.triu_indices(3,1)]  *= -1

        cov_det = np.prod(np.linalg.eigvalsh(cov))
        try:
            assert((self.cov_det(cov) - self.cov_det_ident(self.params[6:13]))/cov_det < 1e-4)
        except:
            print("Determinant formula is wrong...?")
            pdb.set_trace()

        min_axis = 2.0
        #try:
        #    assert(np.min(np.linalg.eigvalsh(cov[:3,:3])) > min_axis**2)
        #except:
        #    print("Minimum positional covariance too small in one direction...")
        #    pdb.set_trace()

        self.icov = np.linalg.inv(cov)
        self.icov_det = np.prod(np.linalg.eigvalsh(self.icov))
        try:
            assert(self.icov_det > 0.0), "negative icov_det"
        except:
            pass
            #print("Negative icov_det")
            #pdb.set_trace()

    def __str__(self):
        return "MVGauss with icov:\n{}\nand icov_det: {}".format(
                    self.icov, self.icov_det)

    def cov_det_ident(self, pars):
        dXinv,dYinv,dZinv,dVinv,xy,xz,yz = pars
        dX = 1/dXinv
        dY = 1/dYinv
        dZ = 1/dZinv
        dV = 1/dVinv
        det = dV**6 * dX**2 * dY**2 * dZ**2 *\
                (1 + 2*xy*xz*yz - xy**2 - xz**2 - yz**2)
        return det

    def cov_det(self, cov):
        return np.prod(np.linalg.eigvalsh(cov))

class Group(MVGaussian):
    """
        Encapsulates the various forms a group model can take
        for example it may be one with fixed parameters and just amplitude
        varying.
    """
    amplitude = None
    age       = None

    def __init__(self, params, amplitude): 
        super(self.__class__,self).__init__(params[:-1])
        self.amplitude = amplitude
        self.age = params[-1] 

    def update_amplitude(self, amplitude):
        self.amplitude = amplitude

# PSEUDO CLASS GroupFitter:
"""
    This class will find the best fitting group models to a set of stars.
    Group models are 6-dimensional multivariate Gausssians which are
    designed to find the instance in time when a set of stars occupied
    the smallest volume
"""

def fit_groups(burnin=100, steps=200, nfree=1, nfixed=0, plotit=True,
             fixed_groups=[],
             infile='results/bp_TGAS2_traceback_save.pkl', pool=None):
    """
    DONE
    returns:
        samples - [NWALKERS x NSTEPS x NPARS] array of all samples
        pos     - the final position of walkers
        lnprob  - [NWALKERS x NSTEPS] array of lnprobablities at each step
    """
    # Data variables
    NDIM    = 6       # number of dimensions for each 'measured' star
    FREE_GROUPS   = []
    FIXED_GROUPS  = []
    # set key values and flags
    FILE_STEM = "gf_bp_{}_{}_{}_{}".format(nfixed, nfree,
                                                burnin, steps)
    NFREE_GROUPS = nfree
    NFIXED_GROUPS = nfixed

    # read in stars from file
    STAR_PARAMS = read_stars(infile)
    print("Work out highest age")
    MAX_AGE = np.max(STAR_PARAMS['times'])
    NSTARS = len(STAR_PARAMS['xyzuvw'])

    # dynamically set initial emcee parameters
    FIXED_GROUPS = [None] * NFIXED_GROUPS
    for i in range(NFIXED_GROUPS):
        FIXED_GROUPS[i] = Group(fixed_groups[i], 1.0)

#    init_free_group_params = [0,0,0,0,0,0,
#                              0.03, 0.03, 0.03,
#                              5,
#                              0, 0, 0,
#                              5.0]
#
#    FREE_GROUPS = [None] * NFREE_GROUPS
#    for i in range(NFREE_GROUPS):
#        FREE_GROUPS[0] = Group(init_group_params, 1.0)

    samples, pos, lnprob =\
             run_fit(burnin, steps, nfixed, nfree, FIXED_GROUPS, STAR_PARAMS,
                     bg=False, pool=pool)

    return samples, pos, lnprob
    # BROKEN!!! NOT ABLE TO DYNAMICALLY CHECK DIFFERENT NUMBER OF GROUPS ATM
    # a way to try and capitalise on groups fitted in the past
    # saved_best = "results/bp_old_best_model_{}_{}".format(NGROUPS,
    #                 NFIXED_GROUPS)
    # try:
    #     print("Trying to open last saved_best")
    #     old_best_lnprob, old_best_model = pickle.load(open(saved_best))
    #     new_best_lnprob = lnprob(init_group_params)
    #     if (old_best_lnprob > new_best_lnprob):
    #         print("-- replacing initial parameters")
    #         init_group_params = old_best_model
    # except:
    #     print("-- unable to open last saved_best")

def read_stars(infile):
    """
    DONE
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
    #Create the inverse covariances to save time.
    xyzuvw_icov = np.linalg.inv(xyzuvw_cov)
    xyzuvw_icov_det = np.linalg.det(xyzuvw_icov)

#   #Store key data in globals
#   STAR_MNS       = xyzuvw
#   STAR_ICOVS     = xyzuvw_icov
#   STAR_ICOV_DETS = xyzuvw_icov_det 

    return dict(stars=stars,times=times,xyzuvw=xyzuvw,xyzuvw_cov=xyzuvw_cov,
                   xyzuvw_icov=xyzuvw_icov,xyzuvw_icov_det=xyzuvw_icov_det)

def lnprior(pars, NFREE_GROUPS, NFIXED_GROUPS, MAX_AGE):
    """
    DONE 
    """
    ngroups = NFREE_GROUPS + NFIXED_GROUPS
    pars = np.array(pars)

    # Generating boolean masks to extract approriate parameters
    # First checking numbers which must be positive (e.g. stddev)
    # the "default_mask" is the mask that would be applied to a single
    # free group. It will be replicated based on the number of free
    # groups currently being fit
    pos_free_mask = 6*[False] + 4*[True] + 3*[False] + [False]

    pos_ampl_mask = [True]
    positive_mask = NFREE_GROUPS * pos_free_mask +\
                            (ngroups -1) * pos_ampl_mask

    # Now generating mask for correlations to ensure in (-1,1) range
    corr_free_mask = 6*[False] + 4*[False] + 3*[True] + [False]
    corr_ampl_mask = [True]
    correlation_mask = NFREE_GROUPS * corr_free_mask +\
                            (ngroups-1) * corr_ampl_mask

    # Generating an age mask to ensure age in (0, MAX_AGE)
    age_free_mask = 6*[False] + 4*[False] + 3*[False] + [True]
    age_ampl_mask = [False]
    age_mask = NFREE_GROUPS * age_free_mask +\
                            (ngroups-1) * age_ampl_mask

    for par in pars[np.where(positive_mask)]:
        if par <= 0:
#           bad_stds += 1
            return -np.inf

    for par in pars[np.where(correlation_mask)]:
        if par <= -1 or par >= 1:
#           bad_corrs += 1
            return -np.inf

    for age in pars[np.where(age_mask)]:
        if age < 0 or age > MAX_AGE:
#           print("Age: {}".format(age))
#           bad_ages += 1
            return -np.inf

    if ngroups > 1:
        amps = pars[-(ngroups-1):]
        if np.sum(amps) > 1:
#           bad_amps += 1
            return -np.inf

    return 0.0

# a function used to set prior on the eigen values
# of the inverse covariance matrix
def eig_prior(char_min, inv_eig_val):
    """
    DONE
    Used to set the prior on the eigen-values of the covariance
    matrix for groups
    """
    eig_val = 1 / inv_eig_val
    prior = eig_val / (char_min**2 + eig_val**2)
    return prior

def lnlike(pars, NFREE_GROUPS, NFIXED_GROUPS, FIXED_GROUPS, star_params):
    """ 
    DONE
    Using the parameters passed in by the emcee run, finds the
    bayesian likelihood that the model defined by these parameters
    could have given rise to the stellar data
    """
    lnlike = 0

    npars_w_age = 14
    free_groups = []
    min_axis = 1.0
    min_v_disp = 0.5
    
    # extract all the amplitudes from parameter list
    amplitudes = pars[NFREE_GROUPS*npars_w_age:]
    assert(len(amplitudes) == NFREE_GROUPS + NFIXED_GROUPS-1),\
                "*** Wrong number of amps"

    # derive the remaining amplitude and append to parameter list
    total_amplitude = sum(amplitudes)
    assert(total_amplitude < 1.0),\
                "*** Total amp is: {}".format(total_amplitude)
    derived_amp = 1.0 - total_amplitude
    pars_len = len(pars)
    pars = np.append(pars, derived_amp)
    amplitudes = pars[NFREE_GROUPS*npars_w_age:]
    assert(len(pars) == pars_len + 1),\
                "*** pars length didn't increase: {}".format(len(pars))
    
    # generate set of Groups based on params and global fixed Groups
    model_groups = [None] * (NFIXED_GROUPS + NFREE_GROUPS)

    # generating the free groups
    for i in range(NFREE_GROUPS):
        group_pars = pars[npars_w_age*i:npars_w_age*(i+1)]
        model_groups[i] = Group(group_pars, amplitudes[i])

    # generating the fixed groups
    for i in range(NFIXED_GROUPS):
        pos = i + NFREE_GROUPS
        #model_groups[pos] = (FIXED_GROUPS[i].params,
        #                           amplitudes[pos], 0)
        FIXED_GROUPS[i].update_amplitude(amplitudes[pos])
        model_groups[pos] = FIXED_GROUPS[i]

    # Handling priors for covariance matrix
    #   if determinant is < 0 then return -np.inf
    #   also incorporates a prior on the eigenvalues being
    #   larger than minimum position/velocity dispersions
    for group in model_groups:
        group_icov_eig = np.linalg.eigvalsh(group.icov)

        # incorporate prior for the eigenvalues
        #   position dispersion
        for inv_eig in group_icov_eig[:3]:
            lnlike += eig_prior(min_axis, inv_eig)

        #   velocity dispersion
        lnlike += eig_prior(min_v_disp, group_icov_eig[3])

        if np.min(group_icov_eig) < 0:
            print("negative determinant...")
#            bad_dets += 1
            return -np.inf

    NSTARS = len(star_params['xyzuvw'])
    ngroups = NFREE_GROUPS + NFIXED_GROUPS
    overlaps = np.zeros((ngroups, NSTARS))

    for i in range(ngroups):
        # prepare group MVGaussian elements
        group_icov = model_groups[i].icov
        group_mn   = model_groups[i].mean
        group_icov_det = model_groups[i].icov_det

        # extract the traceback positions of the stars we're after
        if (model_groups[i].age == 0):
            star_icovs = star_params['xyzuvw_icov'][:,0,:,:]
            star_mns = star_params['xyzuvw'][:,0,:]
            star_icov_dets = star_params['xyzuvw_icov_det'][:,0]
        else:
            star_mns, star_icovs, star_icov_dets =\
                                  interp_icov(model_groups[i].age, star_params)
        
        # use swig to calculate overlaps
        overlaps[i] = overlap.get_overlaps(group_icov, group_mn,
                                           group_icov_det,
                                           star_icovs, star_mns,
                                           star_icov_dets, NSTARS) 
        try:
            assert(np.isfinite(np.sum(overlaps[i])))
        except:
            pdb.set_trace()

    star_overlaps = np.zeros(NSTARS)

    # compile weighted totals of overlaps for each star
    for i in range(NSTARS):
        star_overlaps[i] = np.sum(overlaps[:,i] * amplitudes)

    # return combined product of each star's overlap (or sum of the logs)
#    success += 1
    return np.sum(np.log(star_overlaps))

def lnprobfunc(pars, nfree, nfixed, fixed_groups, star_params):
    """
        DONE
        Compute the log-likelihood for a fit to a group.
        pars are the parameters being fitted for by MCMC 
    """
    # inefficient doing this here, but leaving it for simplicity atm
    max_age = np.max(star_params['times'])

    lp = lnprior(pars, nfree, nfixed, max_age)
    if not np.isfinite(lp):
        #print("Failed priors")
        return -np.inf
    #print("Succeeded")
    return lp + lnlike(pars, nfree, nfixed, fixed_groups, star_params)

def generate_parameter_list(nfixed, nfree, bg=False):
    """
        DONE 
        Generates the initial sample around which the walkers will
        be initialised. This function uses the number of free groups
        and number of fixed groups to dynamically generate a parameter
        list of appropriate length

        bg: Bool, informs the fitter if we are fitting free groups
                  to the background. If we are, the ages of (all) free
                  groups will be fixed at 0.
    """
    # all groups fixed at age = 0
#    if nfixed > NFIXED_GROUPS:
#        print("-- not enough fixed groups provided")
#        nfixed = NFIXED_GROUPS

    init_amp = 1.0 / (nfixed + nfree)
    default_pars = [0,0,0,0,0,0,
                    1./30,1./30,1./30,1./5,
                    0,0,0,
                    5]
    default_sdev = [1,1,1,1,1,1,
                    0.005, 0.005, 0.005, 0.005,
                    0.01,0.01,0.01,
                    0.05] #final 0 is for age

    # If free groups are fitting background set age initval and sdev to 0
    # because emcee generates new samples through linear interpolation
    # between two existing samples, a parameter with 0 init_sdev will not
    # change.
    if bg:
        default_pars[-1] = 0
        default_sdev[-1] = 0

    init_pars = [] + default_pars * nfree + [init_amp]*(nfree+nfixed-1)
    init_sdev = [] + default_sdev * nfree + [0.05]*(nfree+nfixed-1)

    NPAR = len(init_pars)
    NWALKERS = 2*NPAR

    return init_pars, init_sdev, NWALKERS

def run_fit(burnin, steps, nfixed, nfree,
            fixed_groups, star_params, bg=False, pool=None):
    """
        DONE
    """
    # setting up initial params from intial conditions
    init_pars, init_sdev, NWALKERS = generate_parameter_list(nfixed, nfree, bg)
    assert(len(init_pars) == len(init_sdev))
    NPAR = len(init_pars)

    # final parameter is amplitude
    
    p0 = [init_pars+(np.random.random(size=len(init_sdev))- 0.5)*init_sdev
                                            for i in range(NWALKERS)]

    print("In run_fit")
    #pdb.set_trace()
    sampler = emcee.EnsembleSampler(
                        NWALKERS, NPAR, lnprobfunc,
                        args=[nfree, nfixed, fixed_groups, star_params],
                        pool=pool)

    pos, lnprob, state = sampler.run_mcmc(p0, burnin)

    best_chain = np.argmax(lnprob)
    poor_chains = np.where(lnprob < np.percentile(lnprob, 33))
    for ix in poor_chains:
        pos[ix] = pos[best_chain]

    sampler.reset()
    pos,lnprob,rstate = sampler.run_mcmc(pos, steps,
                                              rstate0=state)
    samples = sampler.chain

    # samples is shape [NWALKERS x NSTEPS x NPARS]
    # lnprob is shape [NWALKERS x NSTEPS]
    # pos is the final position of walkers
    return samples, pos, lnprob

def interp_icov(target_time, STAR_PARAMS):
    """
    DONE
    Interpolate in time to get the xyzuvw vector and incovariance matrix.
    """
    times = STAR_PARAMS['times']
    ix = np.interp(target_time, times, np.arange(len(times)))
    ix0 = np.int(ix)
    frac = ix-ix0
    interp_mns       = STAR_PARAMS['xyzuvw'][:,ix0]*(1-frac) +\
                            STAR_PARAMS['xyzuvw'][:,ix0+1]*frac
    interp_icovs     = STAR_PARAMS['xyzuvw_icov'][:,ix0]*(1-frac) +\
                            STAR_PARAMS['xyzuvw_icov'][:,ix0+1]*frac
    interp_icov_dets = STAR_PARAMS['xyzuvw_icov_det'][:,ix0]*(1-frac) +\
                            STAR_PARAMS['xyzuvw_icov_det'][:,ix0+1]*frac
    return interp_mns, interp_icovs, interp_icov_dets
