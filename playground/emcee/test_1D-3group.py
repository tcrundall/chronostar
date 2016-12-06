#!/usr/bin/env python
# test_1D-3group.py

"""
Generates two groups of stars which follow a gaussian distribution
in one dimension at the moment but will extend to 6 dimensions.

Two gaussians are then fitted to the 'data' using MCMC. The gaussians'
properties are extracted from the sampling by taking the modes of the
parameters.

ToDo:
- investigate sofar benign divide by zero error
- introduce consitency with 'weighting' and 'fraction' naming
- get 'align' function to work for arbitrarily many groups
- get plotting to work for arbitrarily many groups
"""

from __future__ import print_function
import emcee
import numpy as np
import math
import argparse
import sys
import pdb
import matplotlib.pyplot as plt
import simulator as sim
from toyfitter import ToyFitter

try:
	xrange
except NameError:
	xrange = range

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--nstars', dest='s', default=50,
												help='[50] number of stars')
parser.add_argument('-f1', '--fraction1', dest='f1', default=0.25,
												help='[0.25] fraction of stars in group 1')
parser.add_argument('-f2', '--fraction2', dest='f2', default=0.5,
												help='[0.5] fraction of stars in group 2')
parser.add_argument('-w', '--nwalkers', dest='w', default=50,
												help='[50] number of walkers')
parser.add_argument('-p', '--steps', dest='p', default=500,
												help='[500] number of sampling steps')
parser.add_argument('-b', '--burnin', dest='b', default=200,
												help='[200] number of burn-in steps')
parser.add_argument('-t', '--plot', dest='plot', action='store_true',
											default=True, help='display and save the plots')
parser.add_argument('-a', '--table', dest='table', action='store_true',
			default=True, help='print a table of stars with their probs')


args = parser.parse_args()

# --- SETTING UP EVERYTHING ----- #

# Setting parameters
print_table = args.table # Display a pretty table with sstars and their groups
plotit = args.plot      # Will plot some pretty graphs at end
initial_help = False    # If walkers are initialised around desired result
nstars = int(args.s)
nwalkers = int(args.w)
first_frac = float(args.f1)
second_frac = float(args.f2)
ndim = 1								# number of phys. dim. being looked at, max 6
ngroups = 3
npar = ngroups*3 - 1		# Number of param. required to define a sample
												# 3 params. per group per dim mean, stdev and weight
burninsteps = int(args.b)	  # Number of burn in steps
samplingsteps = int(args.p)	# Number of sampling steps

# Checks
if (first_frac + second_frac > 1):
  print("Provided fractions must not sum to more than 1")
  sys.exit()

# Useful runtime information
print("Finding a fit for {} stars, with {} walkers for {} steps." \
	.format(nstars, nwalkers, samplingsteps))
if (plotit):
	print("Graphs will be plotted...")
if (print_table):
  print("A table will be printed...")

# Simulating 3 groups as 1-dimensional Gaussian...
# ... with hard coded mean position with pos in pc and vel in km/s
means = [-20.0, 30.0, 100.0]

# ... and some standard deviations
stds = [10.0, 20.0, 5.0]

# --- RUNNING THE FIT --- #
myfit = ToyFitter(nstars, means, stds, first_frac, second_frac,
                  nwalkers, burninsteps, samplingsteps)
myfit.init_stars()
myfit.fit_group()

best_fit = myfit.best_fit
samples = myfit.samples

# Plotting simulated gaussians
if (False):
  xs = np.linspace(-60,120,200)
  
  group1 = (cum_fracs[1] - cum_fracs[0]) * gaussian_eval(xs, means[0], stds[0])
  group2 = (cum_fracs[2] - cum_fracs[1]) * gaussian_eval(xs, means[1], stds[1])
  group3 = (cum_fracs[3] - cum_fracs[2]) * gaussian_eval(xs, means[2], stds[2])
  
  plt.plot(xs, group1)
  plt.plot(xs, group2)
  plt.plot(xs, group3)
  plt.plot(xs, group1 + group2 + group3)
  plt.show()
  
  plt.hist(stars, nstars/5)
  plt.show()

model_mu1  = myfit.best_fit[0]
model_sig1 = myfit.best_fit[1]
model_p1   = myfit.best_fit[2]
model_mu2  = myfit.best_fit[3]
model_sig2 = myfit.best_fit[4]
model_p2   = myfit.best_fit[5]
model_mu3  = myfit.best_fit[6]
model_sig3 = myfit.best_fit[7]
model_p3   = myfit.best_fit[8]

# --- PRINTING RESULTS --- #

# Taking the median of sampled means and sampled stds
# Can compare that to the mean and std on which the stars were
# actually formulated
print(" ____ GROUP 1 _____ ")
print("Modelled mean: {}, modelled std: {}".format(model_mu1, model_sig1))
print("'True' mean: {}, 'true' std: {}".format(means[0], stds[0]))
print("Modelled {}% of the stars, of the true {}%".format(model_p1,
                                                          100*first_frac))

print(" ____ GROUP 2 _____ ")
print("Modelled mean: {}, modelled std: {}".format(model_mu2, model_sig2))
print("'True' mean: {}, 'true' std: {}".format(means[1], stds[1]))
print("Modelled {}% of the stars, of the true {}%".format(model_p2,
                                                          100*second_frac))

print(" ____ GROUP 3 _____ ")
print("Modelled mean: {}, modelled std: {}".format(model_mu3, model_sig3))
print("'True' mean: {}, 'true' std: {}".format(means[2], stds[2]))
print("Modelled {}% of the stars, of the true {}%".format(model_p3,
                                          100*(1-first_frac - second_frac)))

