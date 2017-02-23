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
    # Flags
    PLOTIT = None

    # Data variables
    FILE_STEM = None
    NDIM    = 6       # number of dimensions for each 'measured' star
    NGROUPS = None       # number of groups in the data
    NFIXED_GROUPS = 0
    GROUPS       = []
    FIXED_GROUPS = []
    NSTARS      = None
    STAR_PARAMS = None
    STAR_MNS    = None   # a [NSTARSxNTIMESx6] matrix
    STAR_ICOVS  = None   # a [NSTARSxNTIMESx6x6] matrix
    STAR_ICOV_DETS = None  # a [NSTARS*NTIMES] matrix

    # emcee parameters
    burnin = 100
    steps  = 200
    NWALKERS = 30
    NPAR = 13

    # Fitting variables
    samples    = None
    means      = None  # modelled means [a NGROUPSx6 matrix]
    cov_mats   = None # modelled cov_matrices [a NGROUPSx6x6 matrix]
    weights    = None # the amplitude of each gaussian [a NGROUP matrix]
    best_model = None # best fitting group parameters, same order as 'pars'
    
    def __init__(self, burnin=100, steps=200, ngroups=1, plotit=True,
                 infile='results/bp_TGAS2_traceback_save.pkl'):
        self.PLOTIT = plotit 
        self.burnin = burnin
        self.steps  = steps
        self.NGROUPS = ngroups
        self.STAR_PARAMS = self.read_stars(infile)
        self.NSTARS = len(self.STAR_PARAMS['xyzuvw'])
        init_group_params = [-15.41, -17.22, -21.32, -4.27, -14.39, -5.83,
                              73.34, 51.61, 48.83,
                              7.20,
                             -0.21, -0.09, 0.12]
        self.GROUPS = [None] * self.NGROUPS
        self.GROUPS[0] = Group(init_group_params, 1.0, 0.0)
        self.FILE_STEM = "gf_bp_{}_{}".format(self.burnin, self.steps)

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
        """ Using the parameters passed in by the emcee run, finds the
            bayesian likelihood that the model defined by these parameters
            could have given rise to the stellar data
        """
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

    def fit_groups(self):
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
        self.samples = self.sampler.flatchain

        #Best Model
        best_ix = np.argmax(self.sampler.flatlnprobability)
        self.best_model = self.samples[best_ix]
        print('[' + ",".join(["{0:7.3f}".format(f) for f in self.sampler.flatchain[best_ix]]) + ']')

        self.update_best_model(self.best_model, self.sampler.flatlnprobability[best_ix])
        
        self.write_results()
        if (self.PLOTIT):
            self.make_plots()

    def write_results(self):
        with open("logs/"+self.FILE_STEM+".log", 'w') as f:
            f.write("Log of output from bp with {} burn-in steps, {} sampling steps,\n"\
                        .format(self.burnin, self.steps) )
            f.write("Using starting parameters:\n{}".format(str(self.GROUPS)))
            f.write("\n")

            labels = ["X", "Y", "Z", "U", "V", "W",
                 "dX", "dY", "dZ", "dVel",
                 "xCorr", "yCorr", "zCorr"]
            bf = self.calc_best_params()
            f.write(" _______ BETA PIC MOVING GROUP ________ {starting parameters}\n")
            for i in range(len(labels)):
                f.write("{:8}: {:> 7.2f}  +{:>5.2f}  -{:>5.2f}\t\t\t{:>7.2f}\n".format(
                                                    labels[i],
                                                    bf[i][0], bf[i][1], bf[i][2],
                                                    self.GROUPS[0].params[i]) )

    def calc_best_params(self):
        return np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                        zip(*np.percentile(self.samples, [16,50,84], axis=0))))

    def make_plots(self):
        plt.plot(self.sampler.lnprobability.T)
        plt.title("Lnprob of walkers")
        plt.savefig("plots/lnprob_{}.png".format(self.FILE_STEM))
        plt.clf()

        best_ix = np.argmax(self.sampler.flatlnprobability)
        best_sample = self.samples[best_ix]
        labels = ["X", "Y", "Z", "U", "V", "W",
                 "dX", "dY", "dZ", "dVel",
                 "xCorr", "yCorr", "zCorr"]
        fig = corner.corner(self.samples, truths=best_sample, labels=labels)
        fig.savefig("plots/corner_"+self.FILE_STEM+".png")


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
        file_stem = "results/bp_old_best_model_{}_{}".format(self.NGROUPS, self.NFIXED_GROUPS)
        try:
            old_best_lnprob, old_bet_model = pickle.load(open(file_stem))
            print("Checking old best")
            if (old_best_lnprob < best_lnprob):
                print("Updating with new best: {}".format(best_lnprob))
                pickle.dump((best_lnprob, best_model), open(file_stem, 'w'))
        except:
            print("Storing new best for the first time")
            pickle.dump((best_lnprob, best_model), open(file_stem, 'w'))
