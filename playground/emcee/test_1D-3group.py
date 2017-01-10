#!/usr/bin/env python
# test_1D-3group.py

"""
An example on how to use the ToyFitter class
"""

from __future__ import print_function
import emcee
import numpy as np
import math
import argparse
import sys
import pdb
import matplotlib.pyplot as plt
from toyfitter import ToyFitter

try:
	xrange
except NameError:
	xrange = range

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--nstars',    dest='s', default=100,
												help='[100] number of stars')
parser.add_argument('-f1','--fraction1', dest='f1', default=0.25,
												help='[0.25] fraction of stars in group 1')
parser.add_argument('-f2','--fraction2', dest='f2', default=0.5,
												help='[0.5] fraction of stars in group 2')
parser.add_argument('-w', '--nwalkers',  dest='w', default=50,
												help='[50] number of walkers')
parser.add_argument('-p', '--steps',     dest='p', default=500,
												help='[500] number of sampling steps')
parser.add_argument('-b', '--burnin',    dest='b', default=200,
												help='[200] number of burn-in steps')
parser.add_argument('-t', '--plot',      dest='plot', action='store_true',
											default=True, help='display and save the plots')
parser.add_argument('-a', '--table',     dest='table', action='store_true',
			default=True, help='print a table of stars with their probs')


args = parser.parse_args()

# -------------- SETTING UP EVERYTHING ---------------- #

# Setting parameters
print_table = args.table # Display a pretty table with sstars and their groups
plot_it = args.plot      # Will plot some pretty graphs at end
initial_help = False    # If walkers are initialised around desired result
nstars = int(args.s)
nwalkers = int(args.w)
gr1_frac = float(args.f1)
gr2_frac = float(args.f2)
ndim = 1								# number of phys. dim. being looked at, max 6
ngroups = 3
npar = ngroups*3 - 1		# Number of param. required to define a sample
												# 3 params. per group per dim mean, stdev and weight
burninsteps = int(args.b)	  # Number of burn in steps
samplingsteps = int(args.p)	# Number of sampling steps

# Checks
if (gr1_frac + gr2_frac > 1):
  print("Provided fractions must not sum to more than 1")
  sys.exit()

# Useful runtime information
print("Finding a fit for {} stars, with {} walkers for {} steps with {} burnin." \
	.format(nstars, nwalkers, samplingsteps, burninsteps))
if (plot_it):
	print("Graphs will be plotted...")
if (print_table):
  print("A table will be printed...")

# Simulating 3 groups as 1-dimensional Gaussian with hard coded mean
# positions and some standard deviations
means = [-20.0, 30.0, 100.0]
stds = [10.0, 50.0, 5.0]

# ------------ RUNNING THE FIT --------------- #
myfit = ToyFitter(nstars, means, stds, gr1_frac, gr2_frac,
                  nwalkers, burninsteps, samplingsteps, emcee_detail=True)
myfit.init_stars()
myfit.fit_group()

if (print_table):
  myfit.print_table()
  myfit.print_results()

if (plot_it):
  #myfit.plot_simulated_data()
  myfit.corner_plots()
