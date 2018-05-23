#! /usr/bin/env python -W ignore
"""
This script demos the use of tfgroupfitter. It determines the most likely
origin point of a set of stars assuming a (separate) spherical distribution in
position and velocity space.

It requires the xyzuvw data to be stored in the relative path:
../data/[association-name]_xyzuvw.fits
Results will be stored in:
../results/[association-name]/

cd to scripts/ and Call with:
    nohup mpirun -np [nthreads] python perform_association_fit.py [ass_name] &
or if no mpi installed simply:
    python perform_association_fit.py [ass_name]
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
import sys
from emcee.utils import MPIPool

sys.path.insert(0, '..')

import chronostar.groupfitter as gf

# Initialize the MPI-based pool used for parallelization.
using_mpi = True
mpi_msg = ""    # can't use loggings yet, unclear if appending or rewriting
try:
    pool = MPIPool()
    print("Successfully initialised mpi pool")
except:
    #print("MPI doesn't seem to be installed... maybe install it?")
    print("MPI doesn't seem to be installed... maybe install it?")
    using_mpi = False
    pool=None

try:
    ass_name = sys.argv[1]
except IndexError:
    print(" -------------- INCORRECT USAGE ---------------"
          "  Usage: nohup mpirun -np 19 python [ass_name]"
          " ----------------------------------------------")
    raise

results_dir = "../results/" + ass_name + "/"
xyzuvw_file = '../data/' + ass_name + '_xyzuvw.fits'
best_fit_file = results_dir + "best_fit.npy"

BURNIN_STEPS = 1000
SAMPLING_STEPS = 10000
C_TOL = 0.25

mkpath(results_dir)
logging.basicConfig(
    level=logging.INFO, filemode='a',
    filename=results_dir+'ass_fit.log',
)
logging.info("In preamble")

if using_mpi:
    if not pool.is_master():
        print("One thread is going to sleep")
        # Wait for instructions from the master process.
        pool.wait()

        sys.exit(0)
logging.info("Only one thread is master")

# can optionally initialise fit around approximate coords
#xyzuvw_dict = gf.loadXYZUVW(xyzuvw_file)
#approx_mean = np.mean(xyzuvw_dict['xyzuvw'], axis=0)
#approx_dx   = np.prod(np.std(xyzuvw_dict['xyzuvw'][:,:3], axis=0))**(1./3.)
#approx_dv   = np.prod(np.std(xyzuvw_dict['xyzuvw'][:,3:], axis=0))**(1./3.)
#init_pars = np.hstack((approx_mean, np.log(approx_dx), np.log(approx_dv), 1.0))

logging.info("applying fit")
best_fit, chain, lnprob = gf.fitGroup(
    xyzuvw_file=xyzuvw_file, burnin_steps=BURNIN_STEPS, plot_it=True,
    pool=pool, convergence_tol=C_TOL, save_dir=results_dir, #init_pars=init_pars,
    plot_dir=results_dir, sampling_steps=SAMPLING_STEPS,
)
np.save(best_fit_file, best_fit)

if using_mpi:
    pool.close()
