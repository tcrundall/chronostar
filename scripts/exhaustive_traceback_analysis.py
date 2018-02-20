#! /usr/bin/env python
"""Script to run an exhaustive analysis of synthetic fits and store
their results in a hierarchical directory structure.
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
import sys
import pdb

sys.path.insert(0, '..')

import chronostar.investigator as iv
import chronostar.quadplotter as qp

SAVE_DIR = '../results/synth_results/'
NTIMES = 41
NFIXED_FITS = 21
NSTEPS = 2000

ages = [5, 10, 20]
spreads = [2, 5, 10]
v_disps = [2, 5]
sizes = [25, 50]
precs = ['perf', 'gaia']
"""
ages = [15]
spreads = [5]
v_disps = [5]
sizes   = [25]
precs = ['gaia']
"""

base_group_pars = [
    -80, 80, 50, 10, -20, -5, None, None, None, None,
    0.0, 0.0, 0.0, None, None
]

prec_val = {'perf': 1e-5, 'gaia': 1.0, 'double': 2.0}


def do_something(age, spread, v_disp, size, prec):
    path_name = SAVE_DIR + "{}_{}_{}_{}_{}/".format(
        age, spread, v_disp, size, prec
    )
    mkpath(path_name)
    logging.basicConfig(
        filename=path_name + 'investigator_demo.log',
        level=logging.DEBUG, filemode='w'
    )
    group_pars = list(base_group_pars)
    group_pars[6:9] = [spread, spread, spread]
    group_pars[9] = v_disp
    group_pars[13] = age
    group_pars[14] = size

    times = np.linspace(0, 2 * age, NTIMES)

    sf = iv.SynthFit(init_group_pars=group_pars, save_dir=path_name,
                     times=times, nfixed_fits=NFIXED_FITS, prec=prec_val[prec])
    sf.investigate(period=NSTEPS)
    np.save(path_name + "synthfit.npy", sf)
    qp.quadplot_synth_res(sf, save_dir=path_name)


def do_something_wrapper(scenario):
    print("In wrapper")
    print(".. scenario: {}".format(scenario))
    do_something(*scenario)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        ncpus = int(sys.argv[1])
    else:
        ncpus = 1
    scenarios = product(ages, spreads, v_disps, sizes, precs)
    if ncpus > 1:
        p = Pool(ncpus)
        p.map(do_something_wrapper, scenarios)
    else:
        map(do_something_wrapper, scenarios)

"""    [do_something(age, spread, v_disp, size, prec)
     for age in ages
     for spread in spreads
     for v_disp in v_disps
     for size in sizes
     for prec in precs
     ]

"""
