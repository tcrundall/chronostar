#!/usr/bin/env python
"""
Generating a set of stars which follow a gaussian distribution of UVWXYZ

ToDo:
- assuming no covariance
- extend to look at 2 gaussians
"""

from __future__ import print_function
import emcee
import numpy as np
import math
import sys

try:
	xrange
except NameError:
	xrange = range

# Simulating group as 1-dimensional Gaussian...
ndim = 1
# ... with hard coded mean position with pos in pc and vel in km/s
# means = [35.0, 0.0, 0.0, -10.0, -20.0, -5.0]
means = [35.0]

# ... and some standard deviations

stds = [1.0]
#stds = [3.0, 3.0, 3.0, 1.0, 1.0, 1.0]

# Initialising a set of 50 stars to have UVWXYZ as determined by 
# means and standard devs
nstars = 10 
stars = np.zeros((nstars,ndim))
for i in range(nstars):
	for j in range(ndim):
		stars[i][j] = np.random.normal(means[j], stds[j])

# Defining the probablility distribution to sample
# x encapsulates the mean and std of a proposed model
# i.e. x = [mu, sig]
# the likelihood of the model is the product of probabilities of each star
# given the model, that is evaluate the model gaussian for the given star
# location and product them.
# Since we need the log likelihood, we can take the log of the gaussian at
# each given star and sum them
# Hardcoded to 1 dimension

def gaussian_eval(x, mu, sig):
	result = 1.0/(abs(sig)*math.sqrt(2*math.pi))*np.exp(-(x-mu)**2/(2*sig**2))
	#if (result == 0):
		#print(result)
		#print(x, mu, sig)
	return result

def lnprob(x, stars):
	nstars = stars.size
	mu = x[0]
	sig = x[1]
	sumlnprob = 0
	for i in range(nstars):
		result = np.log(gaussian_eval(stars[i][0], mu, sig))
		if (math.isnan(result)):
			print("Found a NaN!: {} {} {}".format(stars[i][0], mu, sig))
		sumlnprob += result
	return sumlnprob

# We'll sampe with 250 walkers
nwalkers = 250

# Choose an intial set of gaussian parameters for the walkers.
p0 = [np.random.uniform(5,10, [2]) for i in xrange(nwalkers)]

# Initialise the sampler with the chosen specs.
sampler = emcee.EnsembleSampler(nwalkers, 2, lnprob, args=[stars])

# Run 100 steps as burn-in.
burninsteps = 50 
pos, prob, state = sampler.run_mcmc(p0, burninsteps)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position of the burn-in chain, smaple for 1000
# steps.
samplingsteps = 100
sampler.run_mcmc(pos, samplingsteps, rstate0=state)

# Print out the mean acceptance fraction. In general, acceptance_fraction
# has an entry for each walker so, in this case, it is a 250-dimensional
# vector.
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

# Estimate the integrated autocorrelation time for th eitme series in each
# paramter.
print("Autocorrelation time:", sampler.get_autocorr_time())

# Removes the first 100 iterations of each walker and reshapes
# into an ndim*X array where ndim is the number of parameters required
# to specify one position, and X is the number of instances
samples = sampler.chain[:, burninsteps:, :].reshape((-1, ndim))

# Taking average of sampled means and sampled stds
print("Modelled mean: {}, modelled std: {}".format(np.average(samples[0]),
																								np.average(abs(samples[1]))))

# Can compare that to the mean and std on which the stars were
# actually formulated
print("'True' mean: {}, 'true' std: {}".format(means[0], stds[0]))

# Finally, you can plot the porjected histograms of the samples using
# matplotlib as follows
try:
	import matplotlib.pyplot as pl
except ImportError:
	print("Try installing matplotlib to generate some sweet plots...")
else:
	# Plotting all sampled means
	pl.figure(1)
	pl.subplot(211)
	pl.hist(sampler.flatchain[:,0], 100)

	# Plotting all sampled stds
	# Need to take the absolute since emcee samples negative sigmas
	pl.subplot(212)
	sigs = [abs(sig) for sig in sampler.flatchain[:,1]]
	pl.hist(sigs, 100)
	pl.show()
