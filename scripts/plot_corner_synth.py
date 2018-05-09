#! /usr/bin/env python -W ignore
"""
Just a simple hexplotting script for plotting hexplots
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
    age, dX, dV = np.array(sys.argv[1:4], dtype=np.double)
    nstars = int(sys.argv[4])
    precs = sys.argv[5:]
except (IndexError, ValueError):
    print("Usage: python plot_hexplot_synth.py [age] [dX] [dV] [nstars]"
          "[prec1] [prec2] ... ")
    raise

# Setting up file system
rdir = "/data/mash/tcrun/synth_fit/{}_{}_{}_{}/".format(int(age), int(dX),
                                                        int(dV), int(nstars))
if not os.path.isdir(rdir):
    # no access to Tim's RSAA data server, must be working local
    rdir = "../results/synth_fit/{}_{}_{}_{}/".format(int(age), int(dX),
                                                      int(dV), int(nstars))

logging.basicConfig(level=logging.INFO, filename='cornerplotting.log')
print("In preamble")
logging.info("Input arguments: {}".format(sys.argv[1:]))
logging.info("\n"
             "\tage:     {}\n"
             "\tdX:      {}\n"
             "\tdV:      {}\n"
             "\tnstars:  {}\n"
             "\tprecs:   {}".format(
    age, dX, dV, nstars, precs,
))

# since this could be being executed anywhere, need to pass in package
# location
labels = ['X [pc]', 'Y [pc]', 'Z [pc]',
          'U [km/s]', 'V [km/s]', 'W [km/s]',
          'ln(dX)', 'ln(dV)', 'age [Myr]']

for prec in precs:
    logging.info("Plotting for prec {}".format(prec))
    pdir = rdir + prec + '/'
    group = np.load(pdir + 'origins.npy').item()
    group_pars = group.getInternalSphericalPars()
    chain = np.load(pdir + 'final_chain.npy')
    flat = chain.reshape(-1,9)
    plot_file_stem = "{}_{}_{}_{}_{}".format(
        int(age), int(dX), int(dV), nstars, prec
    )
    plt.clf()
    corner.corner(flat, truths=group_pars, labels=labels)
    plt.savefig(pdir + '/corner_' + plot_file_stem + ".pdf")
