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

from distutils.dir_util import mkpath
import numpy as np
import logging
import os
import sys
sys.path.insert(0, '..')
import chronostar.hexplotter as hp

try:
    age, dX, dV = np.array(sys.argv[1:4], dtype=np.double)
    nstars = int(sys.argv[4])
    precs = sys.argv[5:]
except (IndexError, ValueError):
    print("Usage: python plot_hexplot_synth.py [age] [dX] [dV] [nstars]"
          "[prec1] [prec2] ... ")
    raise

rdir = "../results/synth_fit/{}_{}_{}_{}/".format(int(age), int(dX),
                                                  int(dV), int(nstars))

logging.basicConfig(level=logging.INFO, filename='hexplotting.log')
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

for prec in precs:
    try:
        pdir = rdir + prec + '/'
        title = "age {} Myr; dX {} pc; dV {} km/s, nstars {}, prec: {}".format(
            age, dX, dV, nstars, prec
        )
        plot_file_stem = "{}_{}_{}_{}_{}".format(
            int(age), int(dX), int(dV), nstars, prec
        )
        logging.info("Plotting for prec: {}".format(prec))
        hp.dataGatherer(res_dir=pdir, save_dir=pdir,
                        title=title, file_stem=plot_file_stem)
    except IOError:
        logging.info("Couldn't find all the things I needed for prec {}".\
                     format(prec))
