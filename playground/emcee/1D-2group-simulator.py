#!/usr/bin/env python
"""
Generates two groups of stars which follow a gaussian distribution
in one dimension at the moment but will extend to 6 dimensions.

Two gaussians are then fitted to the 'data' using MCMC. The gaussians'
properties are extracted from the sampling by taking the modes of the
parameters.

ToDo:
- have differently sized groups (don't split 50:50)
- generate a list of stars with percentage likelihoods to each group
- test limits with some overlapping groups
"""

from __future__ import print_function
import emcee
import numpy as np
import math

try:
	xrange
except NameError:
	xrange = range

# Pseudo arguments
initial_help = False		# If walkers are initialised around desired result
reorder_samples = True	# If final sample parameters are reordered
nstars = 100 
nwalkers = 150
ndim = 1
npar = 4								# Number of param. required to define a sample
												# 2 params. per group per dimension (mean and stdev)
burninsteps = 100				# Number of burn in steps
samplingsteps = 500		# Number of sampling steps


# Simulating 2 groups as [ndim]-dimensional Gaussian...
# ... with hard coded mean position with pos in pc and vel in km/s
# means = [35.0, 0.0, 0.0, -10.0, -20.0, -5.0]
means1 = [20.0]
means2 = [35.0]

# ... and some standard deviations
stds1 = [1.0]
stds2 = [1.0]
#stds = [3.0, 3.0, 3.0, 1.0, 1.0, 1.0]

# Initialising a set of [nstars] stars to have UVWXYZ as determined by 
# means and standard devs
# [nstars]/2 from one group, [nstars]/2 from the other
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
		sumlnprob += result
	return sumlnprob

# Takes in [nstars][npar] array where each row is a sample and orders each
# sample's parameter sets such that the parameters representing groups are
# listed in ascending order of means
# Hardcoded for 2D
def align_samples(samples):
	new_samples = []
	for sample in samples:
		if sample[0] <= sample[2]:
			new_samples.append(sample)
		else:
			new_sample = [sample[2], sample[3], sample[0], sample[1]]
			new_samples.append(new_sample)
	return np.array(new_samples)


# Choose an intial set of gaussian parameters for the walkers.
# They are 'helped' by being given a similar mean and std
if (initial_help):
	# Walkers are initialised around the vicinity of the groups
	p0 = [
					[np.random.uniform(means1[0]-5,means1[0]+5),
					 np.random.uniform(stds1[0]-0.5, stds1[0]+0.5),
					 np.random.uniform(means2[0]-5,means2[0]+5),
					 np.random.uniform(stds2[0]-0.5, stds2[0]+0.5)]
				for i in xrange(nwalkers)]
else:
	# Walkers aren't initialised around the vicinity of the groups
	# It is important that stds are not initialised to 0
	p0 = [np.random.uniform(5,10, [npar]) for i in xrange(nwalkers)]

# Initialise the sampler with the chosen specs.
sampler = emcee.EnsembleSampler(nwalkers, npar, lnprob, args=[stars])

# Run 100 steps as burn-in.
pos, prob, state = sampler.run_mcmc(p0, burninsteps)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position of the burn-in chain, smaple for 1000
# steps.
sampler.run_mcmc(pos, samplingsteps, rstate0=state)

# Print out the mean acceptance fraction. In general, acceptance_fraction
# has an entry for each walker so, in this case, it is a 250-dimensional
# vector.
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

# Estimate the integrated autocorrelation time for th eitme series in each
# paramter.
print("Autocorrelation time:", sampler.get_autocorr_time())

# Removes the first 100 iterations of each walker and reshapes
# into an npar*X array where npar is the number of parameters required
# to specify one position, and X is the number of instances
if(reorder_samples):
	samples = np.array(align_samples(sampler.chain[:, burninsteps:, :].reshape((-1, npar))))
else:
	samples = np.array(sampler.chain[:, burninsteps:, :].reshape((-1, npar)))

# Taking average of sampled means and sampled stds
print(" ____ GROUP 1 _____ ")
print("Modelled mean: {}, modelled std: {}".format(np.median(samples[:,0]),
																								np.median(abs(samples[:,1]))))

# Can compare that to the mean and std on which the stars were
# actually formulated
print("'True' mean: {}, 'true' std: {}".format(means1[0], stds1[0]))
print(" ____ GROUP 2 _____ ")
print("Modelled mean: {}, modelled std: {}".format(np.median(samples[:,2]),
																								np.median(abs(samples[:,3]))))

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
	mus = [mu for mu in samples[:,0] if (mu > -30) & (mu < 100)]
	pl.hist(mus, nbins)
	pl.title("Means of group 1")

	# Plotting all sampled stds
	# Need to take the absolute since emcee samples negative sigmas
	pl.subplot(222)
	sigs = [abs(sig) for sig in samples[:,1] if abs(sig) < 5]
	pl.hist(sigs, nbins)
	pl.title("Stds of group 1")
	
	pl.subplot(223)
	mus = [mu for mu in samples[:,2] if (mu > -30) & (mu < 100)]
	pl.hist(mus, nbins)
	pl.title("Means of group 2")

	pl.subplot(224)
	sigs = [abs(sig) for sig in samples[:,3] if abs(sig) < 5]
	pl.hist(sigs, nbins)
	pl.title("Stds of group 2")

	pl.savefig("gaussians.png")
	pl.show()
