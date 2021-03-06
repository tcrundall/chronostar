#! /usr/bin/env python -W ignore
"""
This script demos the use of groupfitter, using the traceback
implementation. It determines the most likely origin point of a set of
stars assuming a (separate) spherical distribution in position and
velocity space.

Call with:
    python perform_tb_synth_fit.py [age] [dX] [dV] [nstars] [prec..]
or
    mpirun -np [nthreads] python perform_tb_synth_fit.py [age] [dX] [dV]
    [nstars] [prec..]
where nthreads is the number of threads to be passed into emcee run
"""
from __future__ import division, print_function

import chronostar.synthdata

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
import platform
import sys
from emcee.utils import MPIPool

sys.path.insert(0, '..')

import chronostar.synthdata as syn
import chronostar.traceorbit as torb
import chronostar.retired2.converter as cv
import chronostar.compfitter as gf
import chronostar.coordinate as cc

INIT_WITH_TRUE_ORIGIN = True

prec_val = {'perf': 1e-5, 'half':0.5, 'gaia': 1.0, 'double': 2.0}

BURNIN_STEPS = 500
SAMPLING_STEPS = 5000
C_TOL = 0.25
"""
BURNIN_STEPS = 10
SAMPLING_STEPS = 50
C_TOL = 1.5
"""

print("In preamble")
try:
    age, dX, dV = np.array(sys.argv[1:4], dtype=np.double)
    nstars = int(sys.argv[4])
    precs = sys.argv[5:]
except ValueError:
    print("--------------------- INCORRECT USAGE -----------------------")
    print("nohup mpirun -np 19 python perform_synth_fit.py [age] [dX]"
          " [dV]\n     [nstars] [prec1] [prec2] ... &")
    print("-------------------------------------------------------------")
    raise

# Setting up file system
if platform.system() == 'Linux': # then we are on a RSAA server
    rdir = "/data/mash/tcrun/tb_synth_fit/{}_{}_{}_{}/".format(int(age),
                                                            int(dX),
                                                            int(dV),
                                                            int(nstars))
else: #platform.system() == 'Darwin' # cause no one uses windows....
    rdir = "../results/tb_synth_fit/{}_{}_{}_{}/".format(int(age), int(dX),
                                                      int(dV), int(nstars))
try:
    mkpath(rdir)
except:
    # I guess you're not Tim Crundall... or on an RSAA server
    rdir = "../results/tb_synth_fit/{}_{}_{}_{}/".format(int(age), int(dX),
                                                      int(dV), int(nstars))
    mkpath(rdir)

logging.basicConfig(
    level=logging.INFO, filemode='a',
    filename=rdir + 'trial_synth_fit_log.log'
)

xyzuvw_perf_file     = "perf_xyzuvw.npy"
#result_file          = "result.npy" #not using this.... (?)
group_savefile       = 'origins.npy'
xyzuvw_init_savefile = 'xyzuvw_init.npy'
astro_savefile       = 'astro_table.txt'
xyzuvw_tb_savefile = 'xyzuvw_tb.fits'

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
print("Master should be working in the directory:\n{}".format(rdir))

logging.info("Performing fit with")
logging.info("{} burnin steps".format(BURNIN_STEPS))
logging.info("{} sampling steps".format(SAMPLING_STEPS))
logging.info("{} tolerance".format(C_TOL))
logging.info("In the directory: {}".format(rdir))

# Destination: (inspired by LCC)
mean_now = np.array([50., -100., 0., -10., -20., -5.])
# !!!!!!! ^^^^^ IS WRONG!!!!!!!!
# ^^^^^^ IS LCC IN HELIOCENTRIC COORDINATES. IN LSR COORDINATES IT IS:
mean_now_lsr = cc.convert_helio2lsr(np.array([50., -100., 0.,
                                              -10., -20., -5.]))
# !!!!! WHICH EQUALS [50., -100., 25., 1.1, -7.76, 2.25], much smaller vels
# Calculate appropriate starting point
mean_then = torb.trace_cartesian_orbit(mean_now, -age)
# gather inputs
group_pars = np.hstack((mean_then, dX, dV, age, nstars))

try:
    xyzuvw_now_perf = np.load(rdir+xyzuvw_perf_file)
    origin = np.load(rdir+group_savefile).item()
    logging.info("appending to previous attempt")
except IOError:
    logging.info("Beginning fresh run:")
    logging.info("Input arguments: {}".format(sys.argv[1:]))
    logging.info("\n"
                 "\tage:     {}\n"
                 "\tdX:      {}\n"
                 "\tdV:      {}\n"
                 "\tnstars:  {}\n"
                 "\tprecs:   {}".format(
        age, dX, dV, nstars, precs,
    ))

    # synthesise perfect XYZUVW data
    logging.info("Synthesising data")
    xyzuvw_init, origin = \
        syn.synthesiseXYZUVW(group_pars, form='sphere', return_group=True,
                             xyzuvw_savefile=rdir + xyzuvw_init_savefile,
                             group_savefile=rdir + group_savefile)
    logging.info("Origin has values\n"
                 "\tage:     {}\n"
                 "\tsph_dX:  {}\n"
                 "\tdV:      {}\n"
                 "\tnstars:  {}".format(
        origin.age, origin.sphere_dx, origin.dv, origin.nstars,
    ))
    xyzuvw_now_perf =\
        torb.trace_many_cartesian_orbit(xyzuvw_init, origin.age, single_age=True,
                                        savefile=rdir+xyzuvw_perf_file)

for prec in precs:
    logging.info("Fitting to prec: {}".format(prec))
    pdir = rdir + prec + '/'
    mkpath(pdir)
    # os.chdir(prec)
    try:
        dummy = np.load(pdir+group_savefile)
        logging.info("Precision [{}] already fitted for".format(prec))
    except IOError:
        # convert XYZUVW data into astrometry
        astro_table = chronostar.synthdata.measureXYZUVW(xyzuvw_now_perf, prec_val[prec],
                                                         savefile=pdir+astro_savefile)
        # logging.info("-- Generated astrometry table with errors:")
        # logging.info("Parallax:       {:4} mas".format(ms.GERROR['e_Plx']))
        # logging.info("Radial velocity:{:4} km/s".format(ms.GERROR['e_RV']))
        # logging.info("Proper motion:  {:4} mas/yr".\
        #              format(ms.GERROR['e_pm']))
        star_pars = cv.convertMeasurementsToCartesian(
            astro_table, #savefile=pdir+xyzuvw_conv_savefile
        )
        max_age = int(origin.age) * 2
        ntimes = max_age + 1
        times = np.linspace(0, -max_age, ntimes)
        star_pars_all =\
            torb.generateTracebackFile(star_pars, times,
                                       savefile=pdir+xyzuvw_tb_savefile)
        logging.info("Generated [{}] traceback file".format(prec))

        if INIT_WITH_TRUE_ORIGIN:
            init_pars = origin.getInternalSphericalPars()
        else:
            init_pars = None

        # apply traceforward fitting (with lnprob, corner plots as side
        # effects)
        best_fit, chain, lnprob = gf.fit_comp(
            xyzuvw_dict=star_pars_all, burnin_steps=BURNIN_STEPS, plot_it=True,
            pool=pool, convergence_tol=C_TOL, plot_dir=pdir,
            sampling_steps=SAMPLING_STEPS, save_dir=pdir,
            init_pars=init_pars, traceback=True,
        )
        # store in each directory, for hexplotter
        # also used as a flag to confirm this prec already fitted for
        np.save(pdir+group_savefile, origin)

if using_mpi:
    pool.close()
