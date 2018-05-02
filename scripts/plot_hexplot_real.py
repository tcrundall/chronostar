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
import os
import sys

logging.basicConfig(level=logging.INFO, filename='hexplotting.log')

try:
    package_path = sys.argv[1]
    xyzuvw_file = sys.argv[2]
    if len(sys.argv) > 3:
        res_dir = sys.argv[3] + '/'
    else:
        res_dir = ''
except (ValueError, IndexError):
    print("Usage: python plot_hexplots.py [path-to-chronostar] [xyzuvw_file]"
          "(res_dir)")
    raise
logging.info("Path: {}".format(package_path))
logging.info("xyzuvw_file: {}".format(xyzuvw_file))
logging.info("res_dir: {}".format(res_dir))

# since this could be being executed anywhere, need to pass in package location
sys.path.insert(0, package_path)
try:
    import chronostar.hexplotter as hp
except ImportError:
    #logging.info("Failed to import chronostar package")
    raise

logging.info("Plotting: {}".format(xyzuvw_file))
hp.dataGatherer(res_dir=res_dir, save_dir=res_dir, xyzuvw_file=xyzuvw_file)
