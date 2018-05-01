#! /usr/bin/env python -W ignore
"""
This script demos the use of tfgroupfitter. It determines the most likely
origin point of a set of stars assuming a (separate) spherical distribution in
position and velocity space.

Call with:
python perform_synth_fit.py [age] [dX] [dV] [nstars] [prec..] [path_to_chronostar]
or
mpirun -np [nthreads] python perform_synth_fit.py [age] [dX] [dV] [nstars] [prec..]
    [path_to_chronostar]
where nthreads is the number of threads to be passed into emcee run
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
import pdb
import sys
from emcee.utils import MPIPool

sys.path.insert(0, '..')

import chronostar.synthesiser as syn
import chronostar.traceorbit as torb
import chronostar.converter as cv
import chronostar.measurer as ms
import chronostar.groupfitter as gf
#import chronostar.hexplotter as hp

#xyzuvw_perf_file = "perf_xyzuvw.npy"
results_dir = "../results/twa_core/"
result_file = "result.npy"
#group_savefile = 'origins.npy'
#xyzuvw_init_savefile = 'xyzuvw_init.npy'
#xyzuvw_init_datafile = 'data/sink_init_xyzuvw.npy'
#astro_savefile = 'astro_table.txt'
xyzuvw_file = '../data/twa_core_xyzuvw.fits'
#prec_val = {'perf': 1e-5, 'half':0.5, 'gaia': 1.0, 'double': 2.0}
#data_file = "../data/twa_core_astro.dat"

BURNIN_STEPS = 1000
C_TOL = 0.15

logging.basicConfig(
    level=logging.INFO, filemode='a',
    filename=results_dir+'my_investigator_demo.log',
)
logging.info("In preamble")

# Initialize the MPI-based pool used for parallelization.
using_mpi = True
mpi_msg = ""    # can't use loggings yet, unclear if appending or rewriting
try:
    pool = MPIPool()
    logging.info("Successfully initialised mpi pool")
except:
    #print("MPI doesn't seem to be installed... maybe install it?")
    logging.info("MPI doesn't seem to be installed... maybe install it?")
    using_mpi = False
    pool=None

if using_mpi:
    if not pool.is_master():
        print("One thread is going to sleep")
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)
logging.info("Only one thread is master")

# Performing fit for each precision
logging.info("Fitting")
mkpath(results_dir)
#os.chdir(results_dir)
#np.save(group_savefile, origin) # store in each directory, for hexplotter

logging.info("applying fit")
# apply traceforward fitting (with lnprob, corner plots as side effects)
xyzuvw_dict = gf.loadXYZUVW(xyzuvw_file)
approx_mean = np.mean(xyzuvw_dict['xyzuvw'], axis=0)
approx_dx   = np.prod(np.std(xyzuvw_dict['xyzuvw'][:,:3], axis=0))**(1./3.)
approx_dv   = np.prod(np.std(xyzuvw_dict['xyzuvw'][:,3:], axis=0))**(1./3.)
init_pars = np.hstack((approx_mean, np.log(approx_dx), np.log(approx_dv), 1.0))
best_fit, chain, lnprob = gf.fitGroup(
    xyzuvw_file=xyzuvw_file, burnin_steps=BURNIN_STEPS, plot_it=True,
    pool=pool, convergence_tol=C_TOL, save_dir=results_dir, init_pars=init_pars,
    plot_dir=results_dir
)

if using_mpi:
    pool.close()
