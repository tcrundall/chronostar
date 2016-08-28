#!/usr/bin/env python
"""
Generates two groups of stars which follow a gaussian distribution
in one dimension at the moment but will extend to 6 dimensions.

Two gaussians are then fitted to the 'data' using MCMC. The gaussians'
properties are extracted from the sampling by taking the modes of the
parameters.

ToDo:
- investigate sofar benign divide by zero error
- have differently sized groups (don't split 50:50)
- generate a list of stars with percentage likelihoods to each group
- test limits with some overlapping groups
"""

from __future__ import print_function
import emcee
import numpy as np
import math
import argparse


try:
	xrange
except NameError:
	xrange = range

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--nstars', dest='s', default=50,
												help='number of stars')
parser.add_argument('-f', '--fraction', dest='f', default=0.25,
												help='fraction of stars in group 1')
parser.add_argument('-w', '--nwalkers', dest='w', default=150,
												help='number of walkers')
parser.add_argument('-p', '--steps', dest='p', default=500,
												help='number of sampling steps')
parser.add_argument('-t', '--plot', dest='plot', action='store_true', default=False, \
												help='display and save the plots')
parser.add_argument('-o', '--order', dest='order', action='store_true', default=False, \
												help='reorder samples by mean, only 1D, 2 groups')
parser.add_argument('-a', '--table', dest='table', action='store_true', default=False, \
												help='print a table of stars with their probs')


args = parser.parse_args()

# Setting parameters
print_table = args.table # Display a pretty table with sstars and their groups
plotit = args.plot      # Will plot some pretty graphs at end
initial_help = True		  # If walkers are initialised around desired result
reorder_samples = args.order # If final sample parameters are reordered
nstars = int(args.s)
nwalkers = int(args.w)
fraction = float(args.f)
ndim = 1								# number of phys. dim. being looked at, max 6
npar = 5								# Number of param. required to define a sample
												# 2 params. per group per dimension (mean and stdev)
burninsteps = 100				# Number of burn in steps
samplingsteps = int(args.p)	# Number of sampling steps

# Useful runtime information
print("Finding a fit for {} stars, with {} walkers for {} steps." \
	.format(nstars, nwalkers, samplingsteps))
if (plotit):
	print("Graphs will be plotted...")
if (print_table):
  print("A tabe will be printed...")
if (reorder_samples):
	print("Each sample will have its paramaters made to be ascending...")

# Simulating 2 groups as [ndim]-dimensional Gaussian...
# ... with hard coded mean position with pos in pc and vel in km/s
# means = [35.0, 0.0, 0.0, -10.0, -20.0, -5.0]
means1 = [20.0]
means2 = [40.0]

# ... and some standard deviations
stds1 = [1.0]
stds2 = [1.0]
#stds = [3.0, 3.0, 3.0, 1.0, 1.0, 1.0]

# Initialising a set of [nstars] stars to have UVWXYZ as determined by 
# means and standard devs
# [nstars]/2 from one group, [nstars]/2 from the other
stars = np.zeros((nstars,ndim))
for i in range(int(nstars*fraction)):
	for j in range(ndim):
		stars[i][j] = np.random.normal(means1[j], stds1[j])

for i in range(int(nstars*fraction), nstars):
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

def lnprob(pars, stars):
	nstars = stars.size
	mu1  = pars[0]
	sig1 =abs(pars[1])
	w1   = abs(pars[2])
	mu2  = pars[3]
	sig2 = abs(pars[4])
	sumlnprob = 0
	for i in range(nstars):
		result = np.log(1.0/(1 + w1) * gaussian_eval(stars[i][0], mu1, sig1) +
										1.0/(1.0/w1 + 1) * gaussian_eval(stars[i][0], mu2, sig2) )
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
					[np.random.uniform(means1[0] -5,  means1[0]+5 ),
					 np.random.uniform(stds1[0] -0.5, stds1[0]+0.5),
					 np.random.uniform(2, 3),
					 np.random.uniform(means2[0] -5,  means2[0]+5 ),
					 np.random.uniform(stds2[0] -0.5, stds2[0]+0.5)]
				for i in xrange(nwalkers)]
else:
	# Walkers aren't initialised around the vicinity of the groups
	# It is important that stds are not initialised to 0
	p0 = [np.random.uniform(50,100, [npar]) for i in xrange(nwalkers)]

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

model_mu1  = np.median(samples[:,0])
model_sig1 = np.median(abs(samples[:,1]))
model_w1   = np.median(abs(samples[:,2]))
model_mu2  = np.median(samples[:,3])
model_sig2 = np.median(abs(samples[:,4]))

# Taking average of sampled means and sampled stds
# Can compare that to the mean and std on which the stars were
# actually formulated

print(" ____ GROUP 1 _____ ")
print("Modelled mean: {}, modelled std: {}".format(model_mu1, model_sig1))
print("'True' mean: {}, 'true' std: {}".format(means1[0], stds1[0]))
print("With {}% of the stars".format(100.0/(1+model_w1)))

print(" ____ GROUP 2 _____ ")
print("Modelled mean: {}, modelled std: {}".format(model_mu2, model_sig2))
print("'True' mean: {}, 'true' std: {}".format(means2[0], stds2[0]))
print("With {}% of the stars".format(100.0/(1+1.0/model_w1)))


# Print a list of each star and their predicted group by percentage
# also print the success rate - the number of times a probability > 50 %
# is reported for the correct group
if(print_table):
	print("Star #\tGroup 1\tGroup 2")
	success_cnt = 0.0
	for i, star in enumerate(stars):
		likelihood1 = gaussian_eval(stars[i][0], model_mu1, model_sig1)
		likelihood2 = gaussian_eval(stars[i][0], model_mu2, model_sig2)
		prob1 = likelihood1 / (likelihood1 + likelihood2) * 100
		prob2 = likelihood2 / (likelihood1 + likelihood2) * 100
		if (i < nstars*fraction) & (prob1 > prob2):
			success_cnt += 1.0
		if (i >= nstars*fraction) & (prob1 < prob2):
			success_cnt += 1.0
		print("{}\t{:10.2f}%\t{:10.2f}%".format(i, prob1, prob2))
	print("Success rate of {:6.2f}%".format(success_cnt/nstars * 100))

# Finally, you can plot the porjected histograms of the samples using
# matplotlib as follows
if(plotit):
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
		sigs = [abs(sig) for sig in samples[:,1] if abs(sig) < 30]
		pl.hist(sigs, nbins)
		pl.title("Stds of group 1")
		
		pl.subplot(223)
		mus = [mu for mu in samples[:,3] if (mu > -30) & (mu < 100)]
		pl.hist(mus, nbins)
		pl.title("Means of group 2")

		pl.subplot(224)
		sigs = [abs(sig) for sig in samples[:,4] if abs(sig) < 30]
		pl.hist(sigs, nbins)
		pl.title("Stds of group 2")

		pl.savefig("plots/gaussians.png")
		pl.show()
