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

run_name = sys.argv[1]

try:
    rdir = "/data/mash/tcrun/em_fit/{}/".format(run_name)
    path_msg = "Storing data on mash data server"
    mkpath(rdir)
except (IOError, DistutilsFileError):
    path_msg = ("I'm guessing you're not Tim Crundall..."
                "or not on an RSAA server")
    rdir = "../results/em_fit/{}/".format(run_name)
    mkpath(rdir)

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

logging.info(path_msg)

print("Master should be working in the directory:\n{}".format(rdir))

# Setting up standard filenames
# Partial data generation saved in results/em_fit/[run_name]/synth_data/
sd_dir = rdir + 'synth_data/'
mkpath(sd_dir)
xyzuvw_perf_file     = sd_dir + 'perf_xyzuvw.npy'
groups_savefile      = sd_dir + 'origins.npy'
xyzuvw_init_savefile = sd_dir + 'xyzuvw_init.npy'
astro_savefile       = sd_dir + 'astro_table.txt'
# xyzuvw_conv_savefile = sd_dir + 'xyzuvw_now.fits'

# Final XYZUVW data file stored in chronostar/data/ to replicate
# treatment of real data
xyzuvw_conv_savefile = '../data/{}_xyzuvw.fits'.format(run_name)

mean_now = np.array([50., -100., -0., -10., -20., -5.])
extra_pars = np.array([
    #dX, dV, age, nstars
    [10., 0.5, 20.,  20.],
    [ 2., 1.0,  5., 100.],
    [ 5., 0.7, 10.,  50.],
    # [100., 50.,  1e-5, 1000.],
])
logging.info("Mean (now):\n{}".format(mean_now))
logging.info("Extra pars:\n{}".format(extra_pars))

ERROR = 1.0
ngroups = extra_pars.shape[0]


all_xyzuvw_init = np.zeros((0,6))
all_xyzuvw_now_perf = np.zeros((0,6))

origins = []

logging.info("---------- Generating synthetic data...")
# Calculate the initial parameters for each component that correspond
# to the current day mean of mean_now
for i in range(ngroups):
    logging.info(" generating from group {}".format(i))
    # MANUALLY SEPARATE CURRENT DAY DISTROS IN DIMENSION X
    mean_now_w_offset = mean_now.copy()
    mean_now_w_offset[0] += i * 10

    mean_then = torb.traceOrbitXYZUVW(mean_now, -extra_pars[i,-2],
                                      single_age=True)
    group_pars = np.hstack((mean_then, extra_pars[i]))
    xyzuvw_init, origin = syn.synthesiseXYZUVW(group_pars, sphere=True,
                                               return_group=True,
                                               internal=False)
    origins.append(origin)
    all_xyzuvw_init = np.vstack((all_xyzuvw_init, xyzuvw_init))
    xyzuvw_now_perf = torb.traceManyOrbitXYZUVW(xyzuvw_init,
                                                times=origin.age,
                                                single_age=True)
    all_xyzuvw_now_perf = np.vstack((all_xyzuvw_now_perf, xyzuvw_now_perf))

logging.info(" done")

logging.info("Saving synthetic data...")
np.save(groups_savefile, origins)
np.save(xyzuvw_perf_file, all_xyzuvw_now_perf)
astro_table = ms.measureXYZUVW(all_xyzuvw_now_perf, 1.0,
                               savefile=astro_savefile)

star_pars = cv.convertMeasurementsToCartesian(
    astro_table, savefile=xyzuvw_conv_savefile,
)
em.fitManyGroups(star_pars, ngroups, origins=origins,
                 rdir=rdir, pool=pool,
                 #init_with_origin=True
                 )

if using_mpi:
    pool.close()
