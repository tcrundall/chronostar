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

import logging
import sys

logging.basicConfig(level=logging.INFO, filename='hexplotting.log')

try:
    precs = sys.argv[1:-1]
    package_path = sys.argv[-1]
except ValueError:
    print("Usage: python plot_hexplots.py [prec1] [prec2] [path-to-chronostar]")
    raise
logging.info("Precs: {}".format(precs))
logging.info("Path: {}".format(package_path))

# since this could be being executed anywhere, need to pass in package location
sys.path.insert(0, package_path)
try:
    import chronostar.retired.hexplotter as hp
except ImportError:
    #logging.info("Failed to import chronostar package")
    raise

for prec in precs:
    logging.info("Plotting for prec: {}".format(prec))
    hp.dataGatherer(res_dir=prec+'/', save_dir=prec+'/')
