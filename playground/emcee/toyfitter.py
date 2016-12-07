#!/usr/bin/env python
"""
Generates two groups of stars which follow a gaussian distribution
in one dimension at the moment but will extend to 6 dimensions.

Three gaussians are then fitted to the 'data' using MCMC. The gaussians'
properties are extracted from the sampling by taking the modes of the
parameters.

ToDo:
  trivial:
    - what is xrange?
    - remove unecessary gr_frac  parameters

  important:
    - make a class so state is better encapsulated

  extension:
    - investigate sofar benign divide by zero error
    - get 'align' function to work for arbitrarily many groups
    - get plotting to work for arbitrarily many groups
"""

from __future__ import print_function
import emcee
import numpy as np
import corner
import math
import sys
import pdb
import matplotlib.pyplot as plt


try:
  xrange
except NameError:
  xrange = range

class ToyFitter:
  """
    A class used to find the best fitting group models to a set of stars.
    The stars live in 1 dimension for now. The group models are gaussians.
  """
  # Data variables
  NDIM    = 1       # number of dimensions for each 'measured' star
  NGROUPS = 3       # number of groups in the data
  NSTARS   = None
  gr1_frac = None   # fraction of stars [0.0, 1.0) in the first group
  gr2_frac = None
  gr3_frac = None
  gr_fracs = None
  MEANS = None      # true means of groups in the simulated data
  STDS  = None      # true standard devs of groups in the simulated data
  STARS = None      # simulated data
  CUM_FRACS = None  # cumulative fraction of stars in the groups

  #&
  # Fitting variables
  samples = None
  mdl_means = None  # modelled means
  mdl_stds  = None  # modelled standard deviations
  mdl_fracs = None 
  best_fit  = np.zeros(9) # best fitting group parameters, same order as 'pars'

  # emcee parameters
  burnin = None     # number of burnin steps
  steps = None      # number of sampling steps
  nwalkers = None
  NPAR = 8          # number of parameters required to specify one sample 
  sampler = None    # the emcee.sampler object (initialised in fit_group)

  # some runtime flags
  emcee_detail = False

  def __init__(self, nstars, means, stds, gr1_frac, gr2_frac,
               nwalkers, burnin, steps, emcee_detail=True):
    self.NSTARS = nstars
    self.MEANS = means
    self.STDS = stds
    self.gr1_frac = gr1_frac
    self.gr2_frac = gr2_frac
    self.gr3_frac = 1.0 - gr1_frac - gr2_frac
    self.gr_fracs = [gr1_frac, gr2_frac, 1.0-gr1_frac-gr2_frac]

    self.nwalkers = nwalkers
    self.burnin   = burnin
    self.steps    = steps

    self.emcee_detail = emcee_detail

  # Initialising a set of [nstars] stars to have 'measurments' as determined
  # by means and standard devs
  def init_stars(self):
    self.CUM_FRACS = [0.0, self.gr1_frac, self.gr1_frac+self.gr2_frac, 1.]
    self.STARS = np.zeros(self.NSTARS)
    for i in range(self.NGROUPS):
      start = int(self.NSTARS*self.CUM_FRACS[i])
      end   = int(self.NSTARS*self.CUM_FRACS[i+1])
      for j in range(start, end):
        self.STARS[j] = np.random.normal(self.MEANS[i], self.STDS[i])
  
  # Gaussian helper function
  def gaussian_eval(self, x, mu, sig):
    res = 1.0/(abs(sig)*math.sqrt(2*math.pi))*np.exp(-(x-mu)**2/(2*sig**2))
    return res
  
  # The prior, used to set bounds on the walkers
  def lnprior(self, pars):
    mu1, sig1, w1, mu2, sig2, w2, mu3, sig3 = pars
    if    -200 < mu1 < 200 and 0.0 < sig1 < 50.0 and 5.0 < w1 < 90.0 \
      and -200 < mu2 < 200 and 0.0 < sig2 < 50.0 and 5.0 < w2 < 90.0 \
      and -200 < mu3 < 200 and 0.0 < sig3 < 50.0 and (w2+w1) < 95.0:
      return 0.0
    return -np.inf 
  
  # Defining the probablility distribution to sample
  # x encapsulates the mean and std of a proposed model
  # i.e. x = [mu, sig]
  # the likelihood of the model is the product of probabilities of each star
  # given the model, that is evaluate the model gaussian for the given star
  # location and product them.
  # Since we need the log likelihood, we can take the log of the gaussian at
  # each given star and sum them
  # for each star, we want to find the value of each gaussian at that point
  # and sum them. Every group bar the last has a weighting, the final group's
  # weighting is determined such that the total area under the curve stays 
  # constant (at the moment total area is [ngroups]
  # The awkward multiplicative factor with the weighting is selected
  # so that each factor is between 0 and 1.
  # Currently each star entry only has one value, will eventually extrapolate
  # to many stars
  def lnlike(self, pars):
    #nstars = stars.size
    mu1, sig1, w1, mu2, sig2, w2, mu3, sig3 = pars
    sumlnlike = 0
  
    for i in range(self.NSTARS):
      gaus_sum = ( w1 * self.gaussian_eval(self.STARS[i], mu1, sig1)
                 + w2 * self.gaussian_eval(self.STARS[i], mu2, sig2)
                 + (100-w1-w2) * self.gaussian_eval(self.STARS[i], mu3, sig3) )
  
      sumlnlike += np.log(gaus_sum)
    
    if math.isnan(sumlnlike):
      print("Got a bad'un...")
    return sumlnlike
  
  def lnprob(self, pars):
    lp = self.lnprior(pars)
    if not np.isfinite(lp):
      return -np.inf
    return lp + self.lnlike(pars)
  
  def fit_group(self):
    # initialise walkers, it is important that stds are not initialised to 0
    p0 = [np.random.uniform(10, 60, [self.NPAR]) for i in xrange(self.nwalkers)]
    
    # Initialise the sampler with the chosen specs, run burn-in steps and start
    # sampling
    self.sampler = emcee.EnsembleSampler(self.nwalkers, self.NPAR, self.lnprob)
                                    #args=[self.STARS])
    pos, lnprob, state = self.sampler.run_mcmc(p0, self.burnin)

    # Mike Ireland's code:
    if(self.emcee_detail):
      plt.plot(self.sampler.lnprobability.T)
      plt.title("Is burn in complete?")
      plt.show()
    # This is here to checkout any wayward walkers

    best_chain = np.argmax(lnprob)
    poor_chains = np.where(lnprob < np.percentile(lnprob, 33))
    for ix in poor_chains:
      pos[ix] = pos[best_chain]

    self.sampler.reset()
    self.sampler.run_mcmc(pos, self.steps, rstate0=state)
    
    # Print out the mean acceptance fraction. In general, acceptance_fraction
    # has an entry for each walker
    try:
      print("Mean acceptance fraction:", np.mean(self.sampler.acceptance_fraction))
    except:
      pass
    
    # Estimate the integrated autocorrelation time for the time series in each
    # paramter.
    try:
      print("Autocorrelation time:", self.sampler.get_autocorr_time())
    except:
      pass
    
    # Reshapes each chain into an npar*X array where npar is the number of 
    # parameters required to specify one position, and X is the number of samples
    # "align samples" is used to order groups by their mean within each sample
    self.samples = np.array(self.align_samples(
                               self.sampler.chain[:, :, :].reshape((-1, self.NPAR))))

    self.fit = self.calc_best_fit()
    #pdb.set_trace()
  
    for i in range(9):
      self.best_fit[i] = np.median(self.samples[:,i])

    self.mdl_means  = self.fit[0::3]
    self.mdl_stds   = self.fit[1::3]
    self.mdl_fracs  = self.fit[2::3]

    self.mdl_means2  = self.best_fit[0::3]
    self.mdl_stds2 = self.best_fit[1::3]
    self.mdl_fracs2  = self.best_fit[2::3]

    #pdb.set_trace()

  # Takes in [nstars][npar] array where each row is a sample and orders each
  # sample's parameter sets such that the parameters representing groups are
  # listed in ascending order of means
  # Hardcoded for 3D
  def align_samples(self, samples):
    # Calculating the weighting of the 3rd group, which until now
    # has been implicitly defined through w2 and w1
    w3 = 100 - samples[:,2] - samples[:,5] 
  
    # Append the new weighting to the end of each row
    temp = np.concatenate((samples, w3.T.reshape(-1,1)), axis=1)
  
    # Rearrange each sample into a 3x3 matrix where the extra dimension
    # is for each modelled group
    temp = temp.reshape(-1,3,3)
  
    # Sort each sample by the mean of each modelled group
    temp = np.array([sorted(mat, key=lambda t: t[0]) for mat in temp])
    return temp.reshape(-1,9)

  def calc_best_fit(self):
    return np.array( map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                     zip(*np.percentile(self.samples, [16,50,84], axis=0))) )

  def plot_simulated_data(self):
    xs = np.linspace(-60,120,200)
  
    group1 = self.gr_fracs[0] * self.gaussian_eval(xs, self.MEANS[0], self.STDS[0])
    group2 = self.gr_fracs[1] * self.gaussian_eval(xs, self.MEANS[1], self.STDS[1])
    group3 = self.gr_fracs[2] * self.gaussian_eval(xs, self.MEANS[2], self.STDS[2])
  
    plt.plot(xs, group1)
    plt.plot(xs, group2)
    plt.plot(xs, group3)
    plt.plot(xs, group1 + group2 + group3)
    plt.show()
  
    plt.hist(self.STARS, self.NSTARS/5)
    plt.show()    

  # Print a list of each star and their predicted group by percentage
  # also print the success rate - the number of times a star's membership
  # is correctly calculated 
  def print_table(self):
    print("Star #\tGroup 1\tGroup 2\tGroup 3")
    success_cnt = 0.0
  
    # To calculate the relative likelihoods for a star belonging to a group
    # we simply evaluate the modelled gaussian for each group at the stars
    # 'position'. Normalising these evaluations gives us the porbabilities.
    likelihood = np.zeros(3)
    probs = np.zeros(3)
    for i, star in enumerate(self.STARS):
      for j in range(3):
        likelihood[j] = self.mdl_fracs[j][0] *              \
                        self.gaussian_eval(self.STARS[i],
                                           self.mdl_means[j][0],
                                           self.mdl_stds[j][0])
      prob = 100* likelihood / (np.sum(likelihood))
  
      # We can also test to see if the most probable group was evaluated
      # correctly
      if i<self.NSTARS*self.CUM_FRACS[1] and prob[0]>prob[1] and prob[0]>prob[2]:
        success_cnt += 1.0
      if i>=self.NSTARS*self.CUM_FRACS[1] and i < self.NSTARS*self.CUM_FRACS[2]\
          and prob[0]<prob[1] and prob[1]>prob[2]:
        success_cnt += 1.0
      if i >= self.NSTARS*self.CUM_FRACS[2] \
          and prob[0]<prob[2] and prob[1]<prob[2]:
        success_cnt += 1.0
      print("{}\t{:5.2f}%\t{:5.2f}%\t{:5.2f}%".format(i, prob[0], prob[1], prob[2]))
  
    print("Success rate of {:6.2f}%".format(100*success_cnt/self.NSTARS))

  #&
  def print_results(self):
    for i in range(self.NGROUPS):
      print(" ____ GROUP {} _____ ".format(i+1))
      print("True mean:     {:> 7.2f},  modelled mean:     {:> 7.2f}  +{:>5.2f}  -{:>5.2f}".format(
                        self.MEANS[i], self.mdl_means[i][0],
                        self.mdl_means[i][1], self.mdl_means[i][2]))
      print("True std:      {:> 7.2f},  modelled std:      {:> 7.2f}  +{:>5.2f}  -{:>5.2f}".format(
                        self.STDS[i], self.mdl_stds[i][0],
                        self.mdl_stds[i][1], self.mdl_stds[i][2]))
      print("True fraction: {:> 7.2f}%, modelled fraction: {:> 7.2f}% +{:>5.2f}% -{:>5.2f}%".format(
                        100*self.gr_fracs[i], self.mdl_fracs[i][0],
                        self.mdl_fracs[i][1], self.mdl_fracs[i][2]))
      print()

  def corner_plots(self):
    plt.plot(self.sampler.lnprobability.T)
    plt.title("lnprob of walkers")
    plt.show()
    
    true = [self.MEANS[0], self.STDS[0], 100*self.gr1_frac,
            self.MEANS[1], self.STDS[1], 100*self.gr2_frac,
            self.MEANS[2], self.STDS[2], 100*self.gr3_frac ]
            
    fig = corner.corner(self.samples, labels=["$\mu_1$", "$\sigma_1$", "$w_1$",
                                              "$\mu_2$", "$\sigma_2$", "$w_2$",
                                              "$\mu_3$", "$\sigma_3$", "$w_3$" ],
                        truths=true)
    fig.savefig("plots/triangle.png")
  
  def plot_fits(self):
    # Finally, you can plot the projected histograms of the samples using
    # matplotlib as follows
    if(plotit):
      try:
        import matplotlib.pyplot as plt
      except ImportError:
        print("Try installing matplotlib to generate some sweet plots...")
      else:
        nbins = 500 
        plt.figure(1)
    
        # Plotting all sampled means1
        plt.figure(1)
        plt.subplot(331)
        mus = [mu for mu in samples[:,0] if mu > -150 and mu < 150]
        plt.hist(mus, nbins)
        plt.title("Means of group 1")
    
        # Plotting all sampled stds1
        # Need to take the absolute since emcee samples negative sigmas
        plt.subplot(332)
        sigs = [abs(sig) for sig in samples[:,1] if abs(sig) < 50]
        plt.hist(sigs, nbins)
        plt.title("Stds of group 1")
    
        # Percentages of group 1
        plt.subplot(333)
        plt.hist(samples[:,2], nbins)
        plt.title("Percentages of group 1")
        
        # Means of group 2
        plt.subplot(334)
        mus = [mu for mu in samples[:,3] if mu > -150 and mu < 150]
        plt.hist(mus, nbins)
        plt.title("Means of group 2")
    
        # Stds of group 2
        plt.subplot(335)
        sigs = [abs(sig) for sig in samples[:,4] if abs(sig) < 50]
        plt.hist(sigs, nbins)
        plt.title("Stds of group 2")
    
        # Percentages of group 2
        plt.subplot(336)
        plt.hist(samples[:,5], nbins)
        plt.title("Percentages of group 2")
    
        # Means of group 3 
        plt.subplot(337)
        mus = [mu for mu in samples[:,6] if mu > -150 and mu < 150]
        plt.hist(mus, nbins)
        plt.title("Means of group 3")
    
        # Stds of group 3 
        plt.subplot(338)
        sigs = [abs(sig) for sig in samples[:,7] if abs(sig) < 50]
        plt.hist(sigs, nbins)
        plt.title("Stds of group 3")
    
        # Percentages of group 3
        plt.subplot(339)
        plt.hist(samples[:,8], nbins)
        plt.title("Percentages of group 3")
    
        plt.savefig("plots/gaussians.png")
        plt.show()
