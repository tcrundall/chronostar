#! /usr/bin/env python

try:
    import matplotlib as mpl
    mpl.use('Agg')
except ImportError:
    pass

from distutils.dir_util import mkpath
from distutils.errors import DistutilsFileError
import logging
import numpy as np
import sys
from emcee.utils import MPIPool
sys.path.insert(0, '..')
import chronostar.expectmax as em
import chronostar.groupfitter as gf

try:
    ass_name = sys.argv[1]
except IndexError:
    print(" -------------- INCORRECT USAGE ---------------"
          "  Usage: nohup mpirun -np 19 python [ass_name]"
          " ----------------------------------------------")
    raise

rdir = "../results/em_fit/" + ass_name + "/"
xyzuvw_file = '../data/' + ass_name + '_xyzuvw.fits'
best_fit_file = rdir + "best_fit.npy"

logging.basicConfig(
    level=logging.INFO, filemode='w',
    filename=rdir + 'em.log',
)

# Initialize the MPI-based pool used for parallelization.
using_mpi = True
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
print("Only one thread is master")

#logging.info(path_msg)

print("Master should be working in the directory:\n{}".format(rdir))

logging.info("---------- Generating synthetic data...")
star_pars = gf.loadXYZUVW(xyzuvw_file)

ngroups = 2
em.fitManyGroups(star_pars, ngroups,
                 rdir=rdir, pool=pool,
                 )
if using_mpi:
    pool.close()
