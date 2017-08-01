"""a module for implementing the expectation-maximisation algorithm
in order to fit a multi-gaussian mixture model of moving groups' origins
to a data set of stars tracedback through XYZUVW
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

debug = False

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
            if debug:
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
        if debug:
            try:
                covariance_identity = self.cov_det_ident(self.params[6:13])
                assert((self.cov_det(cov) - covariance_identity)/cov_det < 1e-4)
            except:
                print("Determinant formula is wrong...?")
                pdb.set_trace()

        min_axis = 2.0

        self.icov = np.linalg.inv(cov)
        self.icov_det = np.prod(np.linalg.eigvalsh(self.icov))

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

# def fit_groups(burnin=100, steps=200, nfree=1, nfixed=0, plotit=True,
#              fixed_groups=[], init_free_groups=None, init_free_ages=None,
#              infile='results/bp_TGAS2_traceback_save.pkl', pool=None, bg=False,
#              loc_debug=False, fixed_ages=False):
def fit_groups(infile='results/bp_TGAS2_traceback_save.pkl',
               nfree=2, nbg=0, burnin=100, steps=100):
#             plotit=True,
#             fixed_groups=[], init_free_groups=None, init_free_ages=None,
#             pool=None, bg=False,
#             loc_debug=False, fixed_ages=False):
    """
    Fits a bunch of groups independently to a set of stars paired with a 
    membership list. The fit uses an "Expectation Maximisation" approach which 
    is an iterative approach.
    Parameters
    ----------
        infile - traceback file with (stars, times, xyzuvw, xuzyvw_cov),
            stored as either a pickle or a .fits file
        nfree - number of free groups to be fitted (free being unrestricted
            in age.
        nbg - number of groups to be fitted to the background. Done by fixing
            age to 0.
        burnin - number of burn-in steps to be done by emcee (maybe determined
            internally?)
        steps - number of sampling steps to be done by emcee (maybe determined
            internally?)
    Returns
    -------
        best_fits - a set of the best fitting sample for each group
        memberships - a (nstar x ngroups) numpy array detailing the member-
            ship probabilities.
        (diagnostics) - a set of data to use for plots/margin of errors etc.
    """
#    global debug
#    debug = loc_debug
#
#    # set key values and flags
#    # read in stars from file
#    star_params = read_stars(infile)
#    max_age = np.max(star_params['times'])
#
#    # dynamically set initial emcee parameters
#    FIXED_GROUPS = [None] * nfixed
#    for i in range(nfixed):
#        FIXED_GROUPS[i] = Group(fixed_groups[i], 1.0)
#
#    samples, pos, lnprob =\
#             run_fit(
#                burnin, steps, nfixed, nfree, FIXED_GROUPS,
#                init_free_groups, init_free_ages, star_params,
#                fixed_ages=fixed_ages, bg=bg, pool=pool)
#
#    return samples, pos, lnprob
    return None, None, None

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
        xyzuvw: (nstars,ntimes,6) numpy array, XYZ in pc and UVW in km/s
        xyzuvw_cov: (nstars,ntimes,6,6) numpy array, covariance of xyzuvw
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

def lnprior(pars, nfree, nfixed, max_age):
    """
    """
    ngroups = nfree + nfixed
    pars = np.array(pars)

    # Generating boolean masks to extract approriate parameters
    # First checking numbers which must be positive (e.g. stddev)
    # the "default_mask" is the mask that would be applied to a single
    # free group. It will be replicated based on the number of free
    # groups currently being fit
    base_msk_means  = 6  * [True]  + 4 * [False] + 3 * [False] + [False]
    base_msk_stdevs = 6  * [False] + 4 * [True]  + 3 * [False] + [False]
    base_msk_corrs  = 6  * [False] + 4 * [False] + 3 * [True]  + [False]
    base_msk_ages   = 6  * [False] + 4 * [False] + 3 * [False] + [True]
    base_msk_amps   = 14 * [False]

    mask_means  = nfree * base_msk_means  + (ngroups-1) * [False]
    mask_stdevs = nfree * base_msk_stdevs + (ngroups-1) * [False]
    mask_corrs  = nfree * base_msk_corrs  + (ngroups-1) * [False]
    mask_ages   = nfree * base_msk_ages   + (ngroups-1) * [False]
    mask_amps   = nfree * base_msk_amps   + (ngroups-1) * [True]
    
    if ngroups > 1:
        amps = pars[np.where(mask_amps)]
        if np.sum(amps) > 0.98:
            return -np.inf
        for amp in amps:
            if amp < 0.02:
                return -np.inf

    for stdev in pars[np.where(mask_stdevs)]:
        if stdev <= 0:
            return -np.inf

    for corr in pars[np.where(mask_corrs)]:
        if corr <= -1 or corr >= 1:
            return -np.inf

    for age in pars[np.where(mask_ages)]:
        if age < 0 or age > max_age:
            return -np.inf

    return 0.0

# a function used to set prior on the eigen values
# of the inverse covariance matrix
def eig_prior(char_min, inv_eig_val):
    """
    Used to set the prior on the eigen-values of the covariance
    matrix for groups
    """
    eig_val = 1 / inv_eig_val
    prior = eig_val / (char_min**2 + eig_val**2)
    return prior

def lnlike(pars, star_params, memberships):
    """
    Calculate the log likelihood function of a single group fitted to the
    entire set of stars, with each star's contribution weighted by its
    membership.
    Currently only valid for free groups (those with unfixed age param).

    Parameters
    ----------
    Pars: a list of 14 parameters used to describe the model
        X,Y,Z,U,V,W,dX,dY,dZ,dV,xyC,xzC,yzC,age
    star_params: stellar tracedback positions
    Meberships: a [nstar] array of membership probabilities to group being fit
    Output
    ------
    (float) the log likelihood of the model
    """
    # initial values
    lnlike = 0
    npars = 14
    min_axis = 1.0
    min_v_disp = 0.5
    
    # initialise the covariance matrix etc from pars
    # include a dummy amplitude of 1.0...
    group = Group(pars, 1.0)

    # Handling priors for covariance matrix
    group_icov_eig = np.linalg.eigvalsh(group.icov)

    # Abort fit if any eigenvalue of cov matrix is negative
    if np.min(group_icov_eig) < 0:
        return -np.inf

    # incorporate prior for the eigenvalues
    #   position dispersion
    for inv_eig in group_icov_eig[:3]:
        lnlike += eig_prior(min_axis, inv_eig)

    #   velocity dispersion
    lnlike += eig_prior(min_v_disp, group_icov_eig[3])

    nstars = len(star_params['xyzuvw'])
    # overlaps = np.zeros(nstars)

    # prepare group MVGaussian elements
    group_icov     = group.icov
    group_mn       = group.mean
    group_icov_det = group.icov_det

    # extract the traceback positions of the stars we're after
    if (group.age == 0):
        star_mns = star_params['xyzuvw'][:,0,:]
        star_icovs     = np.linalg.inv(star_params['xyzuvw_cov'][:,0,:,:])
        star_icov_dets = np.linalg.det(star_icovs)
    else:
        star_mns, star_covs =\
                              interp_cov(group.age, star_params)
        star_icovs = np.linalg.inv(star_covs)
        star_icov_dets = np.linalg.det(star_icovs)
    
    # use swig to calculate overlaps
    overlaps = overlap.get_overlaps(group.icov, group.mean,
                                    group.icov_det,
                                    star_icovs, star_mns,
                                    star_icov_dets, nstars)

    # very nasty hack to replace any 'nan' overlaps with a flat 0
    #   ... we'll see if this is fine
    overlaps[np.where(np.isnan(overlaps))] = 0.0

    return lnlike + np.sum(np.log(overlaps**memberships))

def lnprobfunc(pars, nfree, nfixed, fixed_groups, star_params):
    """
        Compute the log-likelihood for a fit to a group.
        pars are the parameters being fitted for by MCMC 
    """
    # inefficient doing this here, but leaving it for simplicity atm
    max_age = np.max(star_params['times'])

    lp = lnprior(pars, nfree, nfixed, max_age)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(pars, nfree, nfixed, fixed_groups, star_params)

def generate_parameter_list(nfixed, nfree, bg=False, init_free_groups=None,
                            init_free_ages=None, fixed_ages=False):
    """
        Generates the initial sample around which the walkers will
        be initialised. This function uses the number of free groups
        and number of fixed groups to dynamically generate a parameter
        list of appropriate length

        bg: Bool, informs the fitter if we are fitting free groups
                  to the background. If we are, the ages of (all) free
                  groups will be fixed at 0.
    """
    npars_in_def = 14

    #init_amp = 1.0 / (nfixed + nfree)
    if nfixed == 0:
        init_amp_free = 1.0 / (nfree)

    else:
        init_amp_free  = 0.2 / (nfree)
        init_amp_fixed = 0.8 / (nfixed)

    default_pars = [0,0,0,0,0,0,
                    1./30,1./30,1./30,1./5,
                    0,0,0,
                    5]

    # default_pars for fitting whole background of large dataset
    # default_pars = [-49.16, -2.57, 38.40, 38.86, 39.71, 40.98,
    #                 1./30,1./30,1./30,1./5,
    #                 0,0,0,
    #                 5]

    default_sdev = [1,1,1,1,1,1,
                    0.005, 0.005, 0.005, 0.005,
                    0.01,0.01,0.01,
                    1.0]

    # If free groups are fitting background set and fix age to 0
    # because emcee generates new samples through linear interpolation
    # between two existing samples, a parameter with 0 init_sdev will not
    # change.
    if bg:
        default_pars[-1] = 0
        default_sdev[-1] = 0

    if fixed_ages:
        default_sdev[-1] = 0

    if nfixed == 0:
        init_pars = [] + default_pars * nfree + [init_amp_free]*(nfree-1)
    else:
        init_pars = [] + default_pars * nfree + [init_amp_free]*nfree +\
                    [init_amp_fixed]*(nfixed-1)
    init_sdev = [] + default_sdev * nfree + [0.05]*(nfree+nfixed-1)

    npar = len(init_pars)
    nwalkers = 2*npar


    # Incorporate initial XYZUVW and ages if present into initialisation
    if init_free_groups is not None:
        for i in range(nfree):
            init_pars[i*npars_in_def : i*npars_in_def + 6] =\
                init_free_groups[i]
    if init_free_ages is not None:
        for i in range(nfree):
            init_pars[i*npars_in_def + 13] =\
                init_free_ages[i]
    return init_pars, init_sdev, nwalkers

def run_fit(burnin, steps, nfixed, nfree, fixed_groups, init_free_groups,
            init_free_ages, star_params, fixed_ages=False, bg=False, pool=None):
    """
    """
    # setting up initial params from intial conditions
    init_pars, init_sdev, nwalkers = generate_parameter_list(
        nfixed, nfree, bg, init_free_groups=init_free_groups,
        init_free_ages=init_free_ages, fixed_ages=fixed_ages)
    if debug:
        assert(len(init_pars) == len(init_sdev))
    npar = len(init_pars)

    # final parameter is amplitude
    
    sampler = emcee.EnsembleSampler(
                        nwalkers, npar, lnprobfunc,
                        args=[nfree, nfixed, fixed_groups, star_params],
                        pool=pool)

    pos = [init_pars+(np.random.random(size=len(init_sdev))- 0.5)*init_sdev
                                            for i in range(nwalkers)]
    nburnins = 4 # arbirtrarily picked this
    burnin_steps_per_run = int(burnin / nburnins)
    state = None
    for i in range(nburnins):
        pos, lnprob, state = sampler.run_mcmc(pos, burnin_steps_per_run, state)

        best_chain = np.argmax(lnprob)
        poor_chains = np.where(lnprob < np.percentile(lnprob, 33))
        for ix in poor_chains:
            pos[ix] = pos[best_chain]
    
        # could have a simple test here to check if all lnprobs are within
        #   5 stdevs of mean lnprob, if not, then burnin some more
        sampler.reset()

    pos,final_lnprob,rstate = sampler.run_mcmc(pos, steps,
                                              rstate0=state)
    samples = sampler.chain
    lnprob  = sampler.lnprobability

    # samples is shape [nwalkers x nsteps x npars]
    # lnprob is shape [nwalkers x nsteps]
    # pos is the final position of walkers
    return samples, pos, lnprob

def interp_cov(target_time, star_params):
    """
    Interpolate in time to get the xyzuvw vector and incovariance matrix.
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

def calc_average_eig(sample):
    """
    Calculates the average eigenvector (proxy for size)
    of the first group listed in the sample
    stdevs must be stored as inverse in sample
    """
    npars_w_age = 14
    group_pars = sample[:npars_w_age]
    assert(group_pars[6] <= 1)
    amplitude=1
    model_group = Group(group_pars, amplitude)
    # careful, nor sure what order the eigen values are returned in
    # seems ok now only because velocity disp is so much smaller than
    # spatial disp
    mean_width = np.mean( np.sqrt((1/np.linalg.eigvalsh(model_group.icov)))[0:3])
    return mean_width
