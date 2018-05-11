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
import logging
import numpy as np
import os
import platform
import sys
sys.path.insert(0, '..')
import chronostar.hexplotter as hp

try:
    dir_name = sys.argv[1]
    ngroups = int(sys.argv[2])
except (IndexError, ValueError):
    print("Usage: python plot_hexplot_em_synth.py [results_dir] [ngroups]")
    raise

# Setting up file system
rdir = "/data/mash/tcrun/em_fit/{}/".format(dir_name.strip('/'))

if not os.path.isdir(rdir):
    # no access to Tim's RSAA data server, must be working local
    rdir = "../results/em_fit/{}/".format(dir_name.strip('/'))

logging.basicConfig(level=logging.INFO, filename=rdir+'em_hexplotting.log')
print("In preamble")
logging.info("Input arguments: {}".format(sys.argv[1:]))

more_iters = True
iter_cnt = 0
while more_iters:
    #try:
    logging.info("Plotting for iter {}".format(iter_cnt))
    idir = rdir + 'iter{}/'.format(iter_cnt)
    hp.dataGathererEM(ngroups=ngroups, iter_count=iter_cnt,
                      save_dir=idir, res_dir=idir,
                      file_stem="{}_".format(dir_name.strip('/'),
                                                  iter_cnt)
                      )
    iter_cnt += 1
    #except IOError:
        #more_iters = False
