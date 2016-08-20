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

try:
	xrange
except NameError:
	xrange = range

# Simulating 2 groups as 1-dimensional Gaussian...
ndim = 1
# ... with hard coded mean position with pos in pc and vel in km/s
# means = [35.0, 0.0, 0.0, -10.0, -20.0, -5.0]
means1 = [35.0]
means2 = [20.0]

# ... and some standard deviations

stds1 = [1.0]
stds2 = [1.0]
#stds = [3.0, 3.0, 3.0, 1.0, 1.0, 1.0]

# Initialising a set of 50 stars to have UVWXYZ as determined by 
# means and standard devs
# 25 from one group, 25 from the other
nstars = 50 
stars = np.zeros((nstars,ndim))
for i in range(nstars/2):
	for j in range(ndim):
		stars[i][j] = np.random.normal(means1[j], stds1[j])

for i in range(nstars/2, nstars):
	for j in range(ndim):
		stars[i][j] = np.random.normal(means2[j], stds2[j])

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
	mu1  = x[0]
	sig1 = x[1]
	mu2  = x[2]
	sig2 = x[3]
	sumlnprob = 0
	for i in range(nstars):
		result = np.log( gaussian_eval(stars[i][0], mu1, sig1) +
										 gaussian_eval(stars[i][0], mu2, sig2) )
		if (math.isnan(result)):
			print("Found a NaN!: {} {} {}".format(stars[i][0], mu, sig))
		sumlnprob += result
	return sumlnprob

# We'll sampe with 250 walkers
nwalkers = 250

# Choose an intial set of gaussian parameters for the walkers.
# They are 'helped' by being given a similar mean and std
initialhelp = True
if (initialhelp):
	# Walkers are initialised around the vicinity of the groups
	p0 = [
					[np.random.uniform(means1[0]-5,means1[0]+5),
					 np.random.uniform(stds1[0]-0.5, stds1[0]+0.5),
					 np.random.uniform(means2[0]-5,means2[0]+5),
					 np.random.uniform(stds2[0]-0.5, stds2[0]+0.5)]
				for i in xrange(nwalkers)]
else:
	# Walkers aren't initialised around the vicinity of the groups
	p0 = [np.random.uniform(5,10, [4]) for i in xrange(nwalkers)]

# Initialise the sampler with the chosen specs.
sampler = emcee.EnsembleSampler(nwalkers, 4, lnprob, args=[stars])

# Run 100 steps as burn-in.
burninsteps = 100 
pos, prob, state = sampler.run_mcmc(p0, burninsteps)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position of the burn-in chain, smaple for 1000
# steps.
samplingsteps = 1000
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
print(" ____ GROUP 1 _____ ")
print("Modelled mean: {}, modelled std: {}".format(np.average(samples[0]),
																								np.average(abs(samples[1]))))

# Can compare that to the mean and std on which the stars were
# actually formulated
print("'True' mean: {}, 'true' std: {}".format(means1[0], stds1[0]))
print(" ____ GROUP 2 _____ ")
print("Modelled mean: {}, modelled std: {}".format(np.average(samples[2]),
																								np.average(abs(samples[3]))))

# Can compare that to the mean and std on which the stars were
# actually formulated
print("'True' mean: {}, 'true' std: {}".format(means2[0], stds2[0]))

# Finally, you can plot the porjected histograms of the samples using
# matplotlib as follows
try:
	import matplotlib.pyplot as pl
except ImportError:
	print("Try installing matplotlib to generate some sweet plots...")
else:
	nbins = 500 
	# Plotting all sampled means1
	pl.figure(1)
	pl.subplot(221)
	pl.hist(sampler.flatchain[:,0], nbins)
	pl.title("Means of group 1")

	# Plotting all sampled stds
	# Need to take the absolute since emcee samples negative sigmas
	pl.subplot(222)
	sigs = [abs(sig) for sig in sampler.flatchain[:,1]]
	pl.hist(sigs, nbins)
	pl.title("Stds of group 1")
	
	pl.subplot(223)
	pl.hist(sampler.flatchain[:,2], nbins)
	pl.title("Means of group 2")

	pl.subplot(224)
	sigs = [abs(sig) for sig in sampler.flatchain[:,3]]
	pl.hist(sigs, nbins)
	pl.title("Stds of group 2")
	pl.show()

	pl.savefig("gaussians.png")
