#! /usr/bin/env python -W ignore
"""
Just a simple script for plotting corner plots of the result of a fit to
an association
"""
from __future__ import division, print_function

try:
    # prevents displaying plots from generation from tasks in background
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not imported")
    pass

import corner
import logging
import numpy as np
import os
import sys
sys.path.insert(0, '..') # even though not explicitly importing anything from
                         # chronostar, still need this to read in instances of
                         # chronostar.Synthesiser.Group
try:
    ass_name = sys.argv[1]
except IndexError:
    print(" -------------- INCORRECT USAGE ---------------"
          "  Usage: nohup mpirun -np 19 python [ass_name]"
          " ----------------------------------------------")
    raise

rdir = "../results/" + ass_name + "/"

logging.basicConfig(level=logging.INFO, filename='cornerplotting.log')
print("In preamble")
logging.info("Input arguments: {}".format(sys.argv[1:]))

# since this could be being executed anywhere, need to pass in package
# location
labels = ['X [pc]', 'Y [pc]', 'Z [pc]',
          'U [km/s]', 'V [km/s]', 'W [km/s]',
          'ln(dX)', 'ln(dV)', 'age [Myr]']

logging.info("Plotting...")
group = np.load(rdir + 'best_fit.npy').item() #!!! CHECK IF THIS IS RIGHT
#group_pars = group.getInternalSphericalPars()
chain = np.load(rdir + 'final_chain.npy')
flat = chain.reshape(-1,9)
lnprob = np.load(rdir + 'final_lnprob.npy')
best_fit = chain[np.unravel_index(np.argmax(lnprob), lnprob.shape)]

plt.clf()
corner.corner(flat, truths=best_fit, labels=labels)
plt.savefig(rdir + '/corner_' + ass_name + ".pdf")
