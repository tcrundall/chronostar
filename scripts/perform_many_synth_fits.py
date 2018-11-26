#! /usr/bin/env python
"""Script to run an exhaustive analysis of synthetic fits and store
their results in a hierarchical directory structure.

include a number to run concurrently e.g.
python tf_exhaustive_traceback_analysis.py 4
"""

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not imported")
    pass

from distutils.dir_util import mkpath
from multiprocessing import Pool
from itertools import product
import logging
import numpy as np
import os
import sys
import pdb

sys.path.insert(0, '..')

#import chronostar.investigator as iv
#import chronostar.quadplotter as qp

#SAVE_DIR = '../results/tf_results/'
#THIS_DIR = os.getcwd()

# best pars: age: 5, spread: 10, v_disp: 2(?), size: 25
"""
# DIFFERENT SERVERS GET DIFFERENT AGE,
# Mash: 5, 15
# Malice: 30 
# Motley : 50
"""
# BASE PARAMETER SET
ages = [30]# [5, 15]#, 30, 50] # motley does 50
spreads = [1, 2] #pc
v_disps = [1, 2] #km/s
sizes = [25, 50, 100] #nstars
precs = ['half', 'gaia', 'double']
labels = ['a', 'b', 'c', 'd']

#precs_string = str(precs).strip("[]").replace(',','').replace("'", '')
precs_string = ' '.join(precs)
prec_val = {'perf': 1e-5, 'half':0.5, 'gaia': 1.0, 'double': 2.0, 'quint':5.0}

def perform_synth_fit_wrapper(scenario):
    logging.info("Fitting: {} ...".format(scenario))
    for label in labels:
        logging.info("--- {} {} ...".format(scenario, label))
        os.system("mpirun -np 4 python perform_synth_fit.py {} {} {} {} "\
            .format(*scenario) + precs_string + " {}".format(label))
        logging.info("--- ... completed: {} {}".format(scenario, label))
    logging.info("... completed: {}".format(scenario))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filemode='a',
                        filename="mash_perform_many_synth_fits.log")
    logging.info("mash is performing fits:\nages: {}\nspreads: {}\n"
                 "v_disps: {}\nsizes: {}\nprecs: {}\n".format(ages, spreads,
                                                              v_disps, sizes,
                                                              precs))
    # read in system arguments
    if len(sys.argv) > 1:
        ncpus = int(sys.argv[1])
    else:
        ncpus = 1
    logging.info("ncpus: {}".format(ncpus))
    scenarios = product(ages, spreads, v_disps, sizes)
    if ncpus > 1:
        p = Pool(ncpus)
        p.map(perform_synth_fit_wrapper, scenarios)
    else:
        map(perform_synth_fit_wrapper, scenarios)

