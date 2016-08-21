#!/usr/bin/env python
"""
Sample code for sampling a multivariate Gaussian using emcee.

"""

from __future__ import print_function
import numpy as np
import emcee
from math import *

try:
    xrange
except NameError:
    xrange = range

# First, define the probability distribution that you would like to sample.
def lnprob(x, mu, icov):
    diff = x-mu
    return -np.dot(diff,np.dot(icov,diff))/2.0

# We'll sample a 1-dimensional Gaussian...
ndim = 1 
# ...with randomly chosen mean position...
means = np.random.rand(ndim)
# ...and a positive definite, non-trivial covariance matrix.
cov  = 0.5-np.random.rand(ndim**2).reshape((ndim, ndim))
cov  = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov  = np.dot(cov,cov)

# Invert the covariance matrix first.
icov = np.linalg.inv(cov)

# We'll sample with 250 walkers.
nwalkers = 250

# Choose an initial set of positions for the walkers.
p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]

# Initialize the sampler with the chosen specs.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[means, icov])

# Run 100 steps as a burn-in.
pos, prob, state = sampler.run_mcmc(p0, 100)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position in the burn-in chain, sample for 1000
# steps.
sampler.run_mcmc(pos, 1000, rstate0=state)

# Print out the mean acceptance fraction. In general, acceptance_fraction
# has an entry for each walker so, in this case, it is a 250-dimensional
# vector.
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

# Estimate the integrated autocorrelation time for the time series in each
# parameter.
print("Autocorrelation time:", sampler.get_autocorr_time())

# Removes the first 100 iterations for each walker and reshapes
# into an ndim*X array where ndim is the number of parameters required
# to specify one position, and X is the number of instances
samples = sampler.chain[:, 100:, :].reshape((-1, ndim))

#samples = sampler.flatchain.flatten()

print("Modelled mean: {}, modelled std: {}".format(np.mean(samples), \
						   np.std(samples)))

print("'True' mean: {}, 'true' std: {}".format(means[0], sqrt(1/icov[0][0])))

# Finally, you can plot the projected histograms of the samples using
# matplotlib as follows (as long as you have it installed).
try:
    import matplotlib.pyplot as pl
except ImportError:
    print("Try installing matplotlib to generate some sweet plots...")
else:
    pl.hist(sampler.flatchain[:,0], 100)
    pl.show()

"""
import matplotlib.pyplot as pl
xl = np.array([0, 10])
for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
    pl.plot(xl, m*xl+b, color="k", alpha=0.1)
pl.plot(xl, m_true*xl+b_true, color="r", lw=2, alpha=0.8)
pl.errorbar(x, y, yerr=yerr, fmt=".k")
"""
