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

SAVE_DIR = '../results/tf_results/'
THIS_DIR = os.getcwd()

# best pars: age: 5, spread: 10, v_disp: 2(?), size: 25
#"""
ages = [5, 10, 20]
spreads = [2, 5, 10]
v_disps = [1, 2, 5]
sizes = [25, 50, 100]
precs = ['perf', 'half', 'gaia', 'double']

"""
ages = [1, 2]
spreads = [5]
v_disps = [2]
sizes   = [25]
precs = ['perf', 'gaia']
"""

precs_string = str(precs).strip("[]").replace(',','').replace("'", '')
prec_val = {'perf': 1e-5, 'half':0.5, 'gaia': 1.0, 'double': 2.0}

def do_something_wrapper(scenario):
    # note that we leave off "prec" in path name
    # this is because the xyzuvw perfect stellar data is shared
    # across the prec scenarios
    path_name = SAVE_DIR + "{}_{}_{}_{}/".format(
        *scenario
    )
    mkpath(path_name)
    os.chdir(path_name)

    os.system("mpirun -np 5 python " +THIS_DIR + "/perform_synth_fit.py {} {} {} {} ".format(
        *scenario) + precs_string + " " + THIS_DIR + "/../"
              )
    os.chdir(THIS_DIR)
    print("Completed: {}".format(scenario))

if __name__ == '__main__':
    # read in system arguments
    if len(sys.argv) > 1:
        ncpus = int(sys.argv[1])
    else:
        ncpus = 1
    scenarios = product(ages, spreads, v_disps, sizes)
    if ncpus > 1:
        p = Pool(ncpus)
        p.map(do_something_wrapper, scenarios)
    else:
        map(do_something_wrapper, scenarios)

