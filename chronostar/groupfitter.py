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
"""

from __future__ import print_function, division

import emcee        # ... duh
import sys          # for MPI
import numpy as np
import matplotlib.pyplot as plt
import pdb          # for debugging
import corner       # for pretty corner plots
import pickle       # for dumping and reading data
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
            assert((self.cov_det(cov) - self.cov_det_ident(self.params[6:13]))/cov_det < 1e-6)
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

    def __init__(self, params, amplitude): 
        super(self.__class__,self).__init__(params[:-1])
        self.amplitude = amplitude
        self.age = params[-1] 

    def update_amplitude(self, amplitude):
        self.amplitude = amplitude

class GroupFitter:
    """
        This class will find the best fitting group models to a set of stars.
        Group models are 6-dimensional multivariate Gausssians which are
        designed to find the instance in time when a set of stars occupied
        the smallest volume
    """
    # Flags
    PLOTIT = None

    # Data variables
    FILE_STEM = None
    NDIM    = 6       # number of dimensions for each 'measured' star
    NGROUPS = None       # number of groups in the data
    NFIXED_GROUPS = 0
    NFREE_GROUPS  = 0
    FREE_GROUPS   = []
    FIXED_GROUPS  = []
    NSTARS      = None
    STAR_PARAMS = None
    STAR_MNS    = None   # a [NSTARSxNTIMESx6] matrix
    STAR_ICOVS  = None   # a [NSTARSxNTIMESx6x6] matrix
    STAR_ICOV_DETS = None  # a [NSTARS*NTIMES] matrix
    MAX_AGE     = None   # the furthest age being tracced back to

    # emcee parameters
    burnin = None
    steps  = None
    NWALKERS = None
    NPAR = None

    # Fitting variables
    samples    = None
    means      = None  # modelled means [a NGROUPSx6 matrix]
    cov_mats   = None # modelled cov_matrices [a NGROUPSx6x6 matrix]
    weights    = None # the amplitude of each gaussian [a NGROUP matrix]
    best_model = None # best fitting group parameters, same order as 'pars'

    # Debugging trackers
    bad_amps  = 0
    bad_ages  = 0
    bad_corrs = 0
    bad_stds  = 0
    bad_eigs  = 0
    bad_dets  = 0
    success   = 0
    
    def __init__(self, burnin=100, steps=200, nfree=1, nfixed=0, plotit=True,
                 fixed_groups=[],
                 infile='results/bp_TGAS2_traceback_save.pkl'):
        # set key values and flags
        self.FILE_STEM = "gf_bp_{}_{}_{}_{}".format(nfixed, nfree,
                                                    burnin, steps)
        self.PLOTIT = plotit 
        self.burnin = burnin
        self.steps  = steps
        self.NFREE_GROUPS = nfree
        self.NFIXED_GROUPS = nfixed

        # read in stars from file
        self.STAR_PARAMS = self.read_stars(infile)
        print("Work out highest age")
        self.MAX_AGE = np.max(self.STAR_PARAMS['times'])
        self.NSTARS = len(self.STAR_PARAMS['xyzuvw'])

        # dynamically set initial emcee parameters
        init_group_params = [-15.41, -17.22, -21.32, -4.27, -14.39, -5.83,
                              73.34, 51.61, 48.83,
                              7.20,
                             -0.21, -0.09, 0.12,
                              0.0]
        init_group_params = [0,0,0,0,0,0,
                              0.03, 0.03, 0.03,
                              5,
                              0, 0, 0,
                              5.0]
        
        self.FIXED_GROUPS = [None] * self.NFIXED_GROUPS
        for i in range(self.NFIXED_GROUPS):
            self.FIXED_GROUPS[i] = Group(fixed_groups[i], 1.0)

        self.FREE_GROUPS = [None] * self.NFREE_GROUPS
        for i in range(self.NFREE_GROUPS):
            self.FREE_GROUPS[0] = Group(init_group_params, 1.0)

        # BROKEN!!! NOT ABLE TO DYNAMICALLY CHECK DIFFERENT NUMBER OF GROUPS ATM
        # a way to try and capitalise on groups fitted in the past
        # saved_best = "results/bp_old_best_model_{}_{}".format(self.NGROUPS,
        #                 self.NFIXED_GROUPS)
        # try:
        #     print("Trying to open last saved_best")
        #     old_best_lnprob, old_best_model = pickle.load(open(saved_best))
        #     new_best_lnprob = self.lnprob(init_group_params)
        #     if (old_best_lnprob > new_best_lnprob):
        #         print("-- replacing initial parameters")
        #         init_group_params = old_best_model
        # except:
        #     print("-- unable to open last saved_best")

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

        #Store key data in globals
        self.STAR_MNS       = xyzuvw
        self.STAR_ICOVS     = xyzuvw_icov
        self.STAR_ICOV_DETS = xyzuvw_icov_det 
    
        return dict(stars=stars,times=times,xyzuvw=xyzuvw,xyzuvw_cov=xyzuvw_cov,
                       xyzuvw_icov=xyzuvw_icov,xyzuvw_icov_det=xyzuvw_icov_det)

    def lnprior(self, pars):
        ngroups = self.NFREE_GROUPS + self.NFIXED_GROUPS
        pars = np.array(pars)

        # Generating boolean masks to extract approriate parameters
        # First checking numbers which must be positive (e.g. stddev)
        # the "default_mask" is the mask that would be applied to a single
        # free group. It will be replicated based on the number of free
        # groups currently being fit
        pos_free_mask = [False, False, False, False, False, False,
                            True, True, True, True, 
                            False, False, False,
                            False]
        pos_ampl_mask = [True]
        positive_mask = self.NFREE_GROUPS * pos_free_mask +\
                                (ngroups -1) * pos_ampl_mask

        # Now generating mask for correlations to ensure in (-1,1) range
        corr_free_mask = [False, False, False, False, False, False,
                            False, False, False, False, 
                            True, True, True,
                            False]
        corr_ampl_mask = [True]
        correlation_mask = self.NFREE_GROUPS * corr_free_mask +\
                                (ngroups-1) * corr_ampl_mask

        # Generating an age mask to ensure age in (0, self.MAX_AGE)
        age_free_mask = [False, False, False, False, False, False,
                            False, False, False, False, 
                            False, False, False,
                            True]
        age_ampl_mask = [False]
        age_mask = self.NFREE_GROUPS * age_free_mask +\
                                (ngroups-1) * age_ampl_mask

        for par in pars[np.where(positive_mask)]:
            if par <= 0:
                self.bad_stds += 1
                return -np.inf

        for par in pars[np.where(correlation_mask)]:
            if par <= -1 or par >= 1:
                self.bad_corrs += 1
                return -np.inf

        for age in pars[np.where(age_mask)]:
            if age < 0 or age > self.MAX_AGE:
                print("Age: {}".format(age))
                self.bad_ages += 1
                return -np.inf

        if ngroups > 1:
            amps = pars[-(ngroups-1):]
            if np.sum(amps) > 1:
                self.bad_amps += 1
                return -np.inf

        return 0.0

    # a function used to set prior on the eigen values
    # of the inverse covariance matrix
    def eig_prior(self, char_min, inv_eig_val):
        """
        Used to set the prior on the eigen-values of the covariance
        matrix for groups
        """
        eig_val = 1 / inv_eig_val
        prior = eig_val / (char_min**2 + eig_val**2)
        return prior

    def lnlike(self, pars):
        """ 
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
        amplitudes = pars[self.NFREE_GROUPS*npars_w_age:]
        assert(len(amplitudes) == self.NFREE_GROUPS + self.NFIXED_GROUPS-1),\
                    "*** Wrong number of amps"

        # derive the remaining amplitude and append to parameter list
        total_amplitude = sum(amplitudes)
        assert(total_amplitude < 1.0),\
                    "*** Total amp is: {}".format(total_amplitude)
        derived_amp = 1.0 - total_amplitude
        pars_len = len(pars)
        pars = np.append(pars, derived_amp)
        amplitudes = pars[self.NFREE_GROUPS*npars_w_age:]
        assert(len(pars) == pars_len + 1),\
                    "*** pars length didn't increase: {}".format(len(pars))
        
        # generate set of Groups based on params and global fixed Groups
        model_groups = [None] * (self.NFIXED_GROUPS + self.NFREE_GROUPS)

        # generating the free groups
        for i in range(self.NFREE_GROUPS):
            group_pars = pars[npars_w_age*i:npars_w_age*(i+1)]
            model_groups[i] = Group(group_pars, amplitudes[i])

        # generating the fixed groups
        for i in range(self.NFIXED_GROUPS):
            pos = i + self.NFREE_GROUPS
            #model_groups[pos] = (self.FIXED_GROUPS[i].params,
            #                           amplitudes[pos], 0)
            self.FIXED_GROUPS[i].update_amplitude(amplitudes[pos])
            model_groups[pos] = self.FIXED_GROUPS[i]

        # Handling priors for covariance matrix
        #   if determinant is < 0 then return -np.inf
        #   also incorporates a prior on the eigenvalues being
        #   larger than minimum position/velocity dispersions
        for group in model_groups:
            group_icov_eig = np.linalg.eigvalsh(group.icov)

            # incorporate prior for the eigenvalues
            #   position dispersion
            for inv_eig in group_icov_eig[:3]:
                lnlike += self.eig_prior(min_axis, inv_eig)

            #   velocity dispersion
            lnlike += self.eig_prior(min_v_disp, group_icov_eig[3])

            if np.min(group_icov_eig) < 0:
                print("negative determinant...")
                self.bad_dets += 1
                return -np.inf

        ngroups = self.NFREE_GROUPS + self.NFIXED_GROUPS
        overlaps = np.zeros((ngroups, self.NSTARS))

        for i in range(ngroups):
            # prepare group MVGaussian elements
            group_icov = model_groups[i].icov
            group_mn   = model_groups[i].mean
            group_icov_det = model_groups[i].icov_det

            # extract the traceback positions of the stars we're after
            if (model_groups[i].age == 0):
                star_icovs = self.STAR_ICOVS[:,0,:,:]
                star_mns = self.STAR_MNS[:,0,:]
                star_icov_dets = self.STAR_ICOV_DETS[:,0]
            else:
                star_mns, star_icovs, star_icov_dets =\
                                      self.interp_icov(model_groups[i].age)
            
            # use swig to calculate overlaps
            overlaps[i] = overlap.get_overlaps(group_icov, group_mn,
                                               group_icov_det,
                                               star_icovs, star_mns,
                                               star_icov_dets, self.NSTARS) 
            try:
                assert(np.isfinite(np.sum(overlaps[i])))
            except:
                pdb.set_trace()

        star_overlaps = np.zeros(self.NSTARS)
    
        # compile weighted totals of overlaps for each star
        for i in range(self.NSTARS):
            star_overlaps[i] = np.sum(overlaps[:,i] * amplitudes)

        # return combined product of each star's overlap (or sum of the logs)
        self.success += 1
        return np.sum(np.log(star_overlaps))
    
    def lnprob(self, pars):
        """Compute the log-likelihood for a fit to a group.
           pars are the parameters being fitted for by MCMC 
        """
        lp = self.lnprior(pars)
        if not np.isfinite(lp):
            #print("Failed priors")
            return -np.inf
        #print("Succeeded")
        return lp + self.lnlike(pars)

    def generate_parameter_list(self, nfixed, nfree):
        """
            Generates the initial sample around which the walkers will
            be initialised. This function uses the number of free groups
            and number of fixed groups to dynamically generate a parameter
            list of appropriate length
        """
        # all groups fixed at age = 0
        if nfixed > self.NFIXED_GROUPS:
            print("-- not enough fixed groups provided")
            nfixed = self.NFIXED_GROUPS

        init_amp = 1.0 / (nfixed + nfree)
        default_pars = [0,0,0,0,0,0,
                        1./30,1./30,1./30,1./5,
                        0,0,0,
                        5]
        default_sdev = [1,1,1,1,1,1,
                        0.005, 0.005, 0.005, 0.005,
                        0.01,0.01,0.01,
                        0.05] #final 0 is for age

        init_pars = [] + default_pars * nfree + [init_amp]*(nfree+nfixed-1)
        init_sdev = [] + default_sdev * nfree + [0.05]*(nfree+nfixed-1)

        self.NPAR = len(init_pars)
        self.NWALKERS = 2*self.NPAR

        return init_pars, init_sdev

    def fit_groups(self, nfixed, nfree):
        # setting up initial params from intial conditions
        init_pars, init_sdev = self.generate_parameter_list(nfixed, nfree)
        assert(len(init_pars) == len(init_sdev))

        # final parameter is amplitude
        
        p0 = [init_pars+(np.random.random(size=len(init_sdev))- 0.5)*init_sdev
                                                for i in range(self.NWALKERS)]

        print("In fit_groups")
        #pdb.set_trace()
        self.sampler = emcee.EnsembleSampler(self.NWALKERS, self.NPAR,
                                             self.lnprob)

        pos, lnprob, state = self.sampler.run_mcmc(p0, self.burnin)

        best_chain = np.argmax(lnprob)
        poor_chains = np.where(lnprob < np.percentile(lnprob, 33))
        for ix in poor_chains:
            pos[ix] = pos[best_chain]
    
        self.sampler.reset()
        self.sampler.run_mcmc(pos, self.steps, rstate0=state)
        self.samples = self.sampler.flatchain

        #Best Model
        best_ix = np.argmax(self.sampler.flatlnprobability)
        self.best_model = self.samples[best_ix]
        print('[' + ",".join(["{0:7.3f}".format(f)\
                for f in self.sampler.flatchain[best_ix]]) + ']')

        self.update_best_model(self.best_model,
                               self.sampler.flatlnprobability[best_ix])

        # # Debugging trackers
        # bad_amps  = 0
        # bad_ages  = 0
        # bad_corrs = 0
        # bad_stds  = 0
        # bad_eigs  = 0
        # bad_dets  = 0
        print("Bad priors")
        print("Amps: {}\nAges: {}\nCorrs: {}\nStds: {}\nEigs: {}\nDets: {}"\
                .format( self.bad_amps, self.bad_ages, self.bad_corrs,
                         self.bad_stds, self.bad_eigs, self.bad_dets))
        print("Success: {}".format(self.success))
        
        tidied_samples = self.tidy_samples(self.samples, self.NFREE_GROUPS,
                                           self.NFIXED_GROUPS)

        # self.write_results()
        if (self.PLOTIT):
            self.make_plots(tidied_samples)

    def tidy_samples(self, samples, nfree, nfixed):
        """
        Currently deriving final weight for plotting reasons.
        Will eventually also convert 1/stds to stds.
        will either convert sampled parameter sto physcial values
        or help "reset" samples after a diverging run...

        Parameters
        ----------
        samples: [(nwalkers*nsteps) x nparams] array of floats
            the flattened array of samples
        """
        # Making room for final weight
        tidied_samples = np.zeros((np.shape(samples)[0],
                                   np.shape(samples)[1]+1))
        # Deriving the remaining weight
        ngroups = nfree + nfixed
        for i, sample in enumerate(samples):
            weights = sample[-(ngroups-1):]
            derived_weight = 1 - np.sum(weights)
            tidied_samples[i] = np.append(sample, derived_weight)

        best_ix = np.argmax(self.sampler.flatlnprobability)
        best_sample = tidied_samples[best_ix]

        #pdb.set_trace()

        return tidied_samples

    def group_metric(self, group1, group2):
        means1 = group1[:6];      means2 = group2[:6]
        stds1  = 1/group1[6:10];  stds2  = 1/group2[6:10]
        corrs1 = group1[10:13];   corrs2 = group2[10:13]
        age1   = group1[13];      age2   = group2[13]

        total_dist = 0
        for i in range(3):
            total_dist += (means1[i] - means2[i])**2 /\
                            (stds1[i]**2 + stds2[i]**2)

        for i in range(3,6):
            total_dist += (means1[i] - means2[i])**2 /\
                            (stds1[3]**2 + stds2[3]**2)

        for i in range(4):
            total_dist += (np.log(stds1[i] / stds2[i]))**2

        for i in range(3):
            total_dist += (corrs1[i] - corrs2[i])**2

        total_dist += (np.log(age1/age2))**2

        return np.sqrt(total_dist)

    def write_results(self):
        """
        not yet made generic... not even being called atm
        """
        with open("logs/"+self.FILE_STEM+".log", 'w') as f:
            f.write("Log of output from bp with {} burn-in steps," ++\
                    "{} sampling steps,\n".format(self.burnin, self.steps) )
            #f.write("Using starting parameters:\n{}".format(str(self.GROUPS)))
            f.write("\n")

            labels = self.generate_labels(self.NFREE_GROUPS,
                                          self.NFIXED_GROUPS)

            bf = self.calc_best_params()
            f.write(" _______ BETA PIC MOVING GROUP ________" ++
                       " {starting parameters}\n")
            for i in range(len(labels)):
                f.write("{:8}: {:> 7.2f}  +{:>5.2f}  -{:>5.2f}\t\t\t{:>7.2f}\n"\
                             .format( labels[i], bf[i][0], bf[i][1], bf[i][2],
                                      self.GROUPS[0].params[i]) )

    def calc_best_params(self):
        return np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                        zip(*np.percentile(self.samples, [16,50,84], axis=0))))

    def make_plots(self, samples):
        """
        Not yet made generic
        """
        plt.plot(self.sampler.lnprobability.T)
        plt.title("Lnprob of walkers")
        plt.savefig("plots/lnprob_{}.png".format(self.FILE_STEM))
        plt.clf()

        #pdb.set_trace()

        # since this function is not yet generic w.r.t. number of groups
        # etc, I've simply hardcoded it to only display the 
        # first 14 parameter fits
        limit = 14
        best_ix = np.argmax(self.sampler.flatlnprobability)
        best_sample = samples[best_ix]
        labels = self.generate_labels(self.NFREE_GROUPS, self.NFIXED_GROUPS)
        fig = corner.corner(samples[:,:limit], truths=best_sample[:limit],
                             labels=labels[:limit])
        fig.savefig("plots/corner_"+self.FILE_STEM+".png")

    def generate_labels(self, nfree, nfixed):
        """
        Dynamically generates a set of labels for an arbitrary number
        of free and fixed groups.
        e.g. 1 free and 1 fixed:
        ["X0", "Y0", "Z0', ...
        ... "age0", "weight0", "weight1"]
        weight's will all appear consecutively at the very end. Note that
        there will be a "weight" for the derived weight for the final group
        """
        base_label = ["X", "Y", "Z", "U", "V", "W",
                      "dX", "dY", "dZ", "dVel",
                      "xCorr", "yCorr", "zCorr", "age"]

        labels = []
        for i in range(nfree):
            labels += [lb + str(i) for lb in base_label]

        # includes a label for the derived weight
        for i in range(nfree + nfixed):
            labels += ["weight" + str(i)]
        return labels

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

    def update_best_model(self, best_model, best_lnprob):
        file_stem = "results/bp_old_best_model_{}_{}"\
                        .format(self.NGROUPS, self.NFIXED_GROUPS)
        try:
            old_best_lnprob, old_best_model = pickle.load(open(file_stem))
            print("Checking old best")
            if (old_best_lnprob < best_lnprob):
                print("Updating with new best: {}".format(best_lnprob))
                pickle.dump((best_lnprob, best_model), open(file_stem, 'w'))
        except:
            print("Storing new best for the first time")
            pickle.dump((best_lnprob, best_model), open(file_stem, 'w'))
