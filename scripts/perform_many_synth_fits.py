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
ages = [5, 15, 30, 50]  #Myr
spreads = [1, 2, 5, 10] #pc
v_disps = [1, 2, 5, 10] #km/s
sizes = [25, 50, 100, 200] #nstars
precs = ['perf', 'half', 'gaia', 'double']

ages = [30]
spreads = [5]
v_disps = [2]
sizes   = [25, 100]
precs = ['perf', 'gaia']
"""
"""
# DIFFERENT SERVERS GET DIFFERENT AGE,
# Mash: 5
# Malice: 15
# Motley : 30, 50
ages = [5, 15, 30, 50]
spreads = [1, 5] #pc
v_disps = [2, 10] #km/s
sizes = [25, 100] #nstars
precs = ['perf', 'half', 'gaia']
"""
# EVEN PARAMETERS INITED
ages = [5, 15, 30, 50]
spreads = [2, 10] #pc
v_disps = [1, 5] #km/s
sizes = [50, 200] #nstars
precs = ['perf', 'half', 'gaia']


#precs_string = str(precs).strip("[]").replace(',','').replace("'", '')
precs_string = ' '.join(precs)
prec_val = {'perf': 1e-5, 'half':0.5, 'gaia': 1.0, 'double': 2.0}

def perform_synth_fit_wrapper(scenario):
    logging.info("Fitting: {}".format(scenario))
    os.system("mpirun -np 7 python perform_synth_fit.py {} {} {} {} "\
        .format(*scenario) + precs_string)
    #os.system("python perform_synth_fit.py {} {} {} {} " \
    #          .format(*scenario) + precs_string)
    logging.info("Completed: {}".format(scenario))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filemode='w',
                        filename="motley_perform_many_synth_fits.log")
    logging.info("motley is performing many fits:\nages: {}\nspreads: {}\n"
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

