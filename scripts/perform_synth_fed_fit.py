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
import platform
from emcee.utils import MPIPool
import sys
sys.path.insert(0, '..')

import chronostar.synthesiser as syn
import chronostar.traceorbit as torb
import chronostar.converter as cv
import chronostar.measurer as ms
import chronostar.groupfitter as gf
import chronostar.datatool as dt
# import chronostar.retired.hexplotter as hp

#base_sphere_pars = [100, -80, 40, -7, -17, -7, None, None, None, None]

INIT_WITH_TRUE_ORIGIN = False

prec_val = {'perf':1e-5, 'half':0.5, 'gaia':1.0, 'double':2.0, 'quint':5.0}


BURNIN_STEPS = 1000
SAMPLING_STEPS = 1000
C_TOL = 0.25

print("In preamble")

# stops plots popping up as they are created, mayhaps too late if only
# put here....

try:
    #age, dX, dV = np.array(sys.argv[1:4], dtype=np.double)
    age = float(sys.argv[1])
    precs = sys.argv[2:-1]
    if precs[-1] not in prec_val.keys():
        label = precs.pop(-1)
    else:
        label = None
except ValueError:
    print("--------------------- INCORRECT USAGE --------------------------")
    print("nohup mpirun -np 19 python perform_synth_fit.py [age] "
          "[prec1]\n [prec2] ... [label] &")
    print("----------------------------------------------------------------")
    raise

# Setting up file system
if platform.system() == 'Linux': # then we are on a RSAA server
    rdir = "/data/mash/tcrun/fed_fits/{}/".format(int(age))
    if label:
        rdir = rdir[:-1] + '_{}/'.format(label)

else: #platform.system() == 'Darwin' # cause no one uses windows....
    rdir = "../results/fed_fits/{}/".format(int(age))
    if label:
        rdir = rdir[:-1] + '_{}/'.format(label)
try:
    mkpath(rdir)
except:
    # I guess you're not Tim Crundall... or on an RSAA server
    rdir = "../results/fed_fits/{}/".format(int(age))
    if label:
        rdir = rdir[:-1] + '_{}/'.format(label)
    mkpath(rdir)

logging.basicConfig(
    level=logging.INFO, filemode='a',
    filename=rdir + 'fed_synth_fit.log'
)

xyzuvw_perf_file = 'perf_xyzuvw.npy'
# result_file = 'result.npy'
group_savefile = 'origins.npy'
#xyzuvw_init_savefile = 'xyzuvw_init.npy'
xyzuvw_init_datafile = '../data/sink_init_xyzuvw.npy'
astro_savefile = 'astro_table.txt'
xyzuvw_conv_savefile = 'xyzuvw_now.fits'


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

## Destination: (inspired by LCC)
mean_now_lsr = np.array([50., -100., 25., 1.1, -7.76, 2.25])

# Calculate appropriate starting point
mean_then = torb.traceOrbitXYZUVW(mean_now_lsr, -age)
# gather inputs
# group_pars = np.hstack((mean_then, dX, dV, age, nstars))

# Setting up perfect current xyzuvw values
try:
    xyzuvw_now_perf = np.load(rdir+xyzuvw_perf_file)
    # origin = np.load(group_savefile)
    origin = dt.loadGroups(rdir+group_savefile)[0]
    logging.basicConfig(
        level=logging.INFO, filemode='a',
        filename='my_investigator_demo.log',
    )
    logging.info("appending to previous attempt")
except IOError:
    logging.basicConfig(
        level=logging.INFO, filemode='w',
        filename='my_investigator_demo.log',
    )
    logging.info("Beginning fresh run:")
    logging.info("Input arguments: {}".format(sys.argv[1:]))
    logging.info("\n"
                 "\tage:     {}\n"
                 "\tprecs:   {}".format(
        age, precs,
    ))

    # synthesise perfect XYZUVW data
    logging.info("Synthesising data")
    xyzuvw_init = np.load(xyzuvw_init_datafile)

    # incorporating offset so the group is measured at a consistent distance
    # approx_final_pos = np.array([80, 100, -20])


    # dist_travelled = 50. #pc
    # vel_offset = np.array([dist_travelled/age, dist_travelled/age, 0.])
    # offset = np.append([0,0,0], vel_offset)
    # offset[0] -= age * offset[3]
    # offset[1] -= age * offset[4]
    # offset[2] -= age * offset[5]


    xyzuvw_init += mean_then
    np.save(rdir+'xyzuvw_init_offset.npy', xyzuvw_init)
    # xyzuvw_init[:,:3] += approx_final_pos

    # fit an approximate Gaussian to initial distribution for reference
    mean = np.mean(xyzuvw_init, axis=0)
    dx, dy, dz = np.std(xyzuvw_init[:,:3], axis=0)
    dv = np.prod(np.std(xyzuvw_init[:,3:], axis=0))**(1./3.)
    group_pars = np.hstack((mean, dx, dy, dz, dv, 0., 0., 0., age))
    origin = syn.Group(group_pars, internal=False, sphere=False, starcount=False)
    np.save(group_savefile, origin)

    xyzuvw_now_perf =\
        torb.traceManyOrbitXYZUVW(xyzuvw_init, age, single_age=True,
                                  savefile=rdir+xyzuvw_perf_file)

if not using_mpi:
    logging.info("MPI available! - call this with e.g. mpirun -np 19"
                 " python perform_synth_fit.py")

# Performing fit for each precision
for prec in precs:
    logging.info("Fitting to prec: {}".format(prec))
    pdir = rdir + prec + '/'
    mkpath(pdir)
    np.save(pdir+group_savefile, origin) # store in each directory, for hexplotter
    try:
        best_group = dt.loadGroups(pdir + 'final_groups.npy')
        logging.info("Precision [{}] already fitted for".format(prec))
    except IOError:
        # convert XYZUVW data into astrometry
        astro_table = ms.measureXYZUVW(xyzuvw_now_perf, prec_val[prec],
                                       savefile=pdir+astro_savefile)
        star_pars =\
            cv.convertMeasurementsToCartesian(astro_table,
                                              savefile=pdir+xyzuvw_conv_savefile)
        logging.info("Generated [{}] traceback file".format(prec))

        # apply traceforward fitting (with lnprob, corner plots as side effects)
        best_fit, chain, lnprob = gf.fitGroup(
            xyzuvw_dict=star_pars, burnin_steps=BURNIN_STEPS, plot_it=True,
            pool=pool, convergence_tol=C_TOL
        )
        best_group = syn.Group(best_fit, sphere=True,
                               internal=True, star_count=False)
        np.save(pdir + 'final_groups.npy')
        #hp.dataGatherer(save_dir=prec)


if using_mpi:
    pool.close()
