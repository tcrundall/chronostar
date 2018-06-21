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
    print(" ---------------- INCORRECT USAGE ---------------\n"
          "  Usage: nohup mpirun -np 19 python\n"
          "         perform_em_association_fit.py [ass_name]\n"
          " ------------------------------------------------")
    raise


try:
    rdir = "/data/mash/tcrun/em_fit/{}/".format(ass_name.strip('/'))
    path_msg = "Storing data on mash data server"
    mkpath(rdir)
except (IOError, DistutilsFileError):
    path_msg = ("I'm guessing you're not Tim Crundall..."
                "or not on an RSAA server")
    rdir = "../results/em_fit/{}/".format(ass_name)
    if rdir[-1] != '/':
        rdir += '/'
    mkpath(rdir)

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

star_pars = gf.loadXYZUVW(xyzuvw_file)

ngroups = 2
logging.info("Everythign loaded, about to fit with {} components"\
    .format(ngroups))
em.fitManyGroups(star_pars, ngroups,
                 rdir=rdir, pool=pool,
                 )
if using_mpi:
    pool.close()
