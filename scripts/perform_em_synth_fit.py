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
import chronostar.synthesiser as syn
import chronostar.traceorbit as torb
import chronostar.converter as cv
import chronostar.measurer as ms
import chronostar.expectmax as em

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


dir_name = sys.argv[1]

try:
    rdir = "/data/mash/tcrun/em_fit/" + dir_name
    if rdir[-1] != '/':
        rdir += '/'
    mkpath(rdir)
except (IOError, DistutilsFileError):
    print("I'm guessing you're not Tim Crundall..."
          "or not on an RSAA server")
    rdir = "../results/em_fit/" + dir_name
    if rdir[-1] != '/':
        rdir += '/'
    mkpath(rdir)

print("Master should be working in the directory:\n{}".format(rdir))
#os.chdir(res_dir)

logging.basicConfig(
    level=logging.INFO, filemode='w',
    filename=rdir + 'em.log',
)

# Setting up standard filenames
xyzuvw_perf_file     = 'perf_xyzuvw.npy'
groups_savefile      = 'origins.npy'
xyzuvw_init_savefile = 'xyzuvw_init.npy'
astro_savefile       = 'astro_table.txt'
xyzuvw_conv_savefile = 'xyzuvw_now.fits'

mean_now = np.array([50., -100., -0., -10., -20., -5.])
extra_pars = np.array([
    #dX, dV, age, nstars
    [10., 3., 10., 20.],
    [10., 5.,  7., 100.],
])
logging.info("Mean (now):\n{}".format(mean_now))
logging.info("Extra pars:\n{}".format(extra_pars))

ERROR = 1.0
ngroups = extra_pars.shape[0]

all_group_pars = np.zeros((0,10))

all_xyzuvw_init = np.zeros((0,6))
all_xyzuvw_now_perf = np.zeros((0,6))

origins = []

for i in range(ngroups):
    mean_then = torb.traceOrbitXYZUVW(mean_now, -extra_pars[i,-2],
                                      single_age=True)
    group_pars = np.hstack((mean_then, extra_pars[i]))
    xyzuvw_init, group =\
        syn.synthesiseXYZUVW(group_pars, sphere=True, return_group=True)
    all_xyzuvw_init = np.vstack((all_xyzuvw_init, xyzuvw_init))

    xyzuvw_now_perf = torb.traceManyOrbitXYZUVW(xyzuvw_init, group.age,
                                                single_age=True)
    all_xyzuvw_now_perf =\
        np.vstack((all_xyzuvw_now_perf, xyzuvw_now_perf))
    origins.append(group)

np.save(rdir+groups_savefile, origins)
np.save(rdir+xyzuvw_perf_file, all_xyzuvw_now_perf)
astro_table = ms.measureXYZUVW(all_xyzuvw_now_perf, 1.0,
                               savefile=rdir+astro_savefile)

star_pars = cv.convertMeasurementsToCartesian(
    astro_table, savefile=rdir+xyzuvw_conv_savefile,
)
em.fitManyGroups(star_pars, ngroups, rdir=rdir, pool=pool)
