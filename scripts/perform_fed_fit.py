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

#base_sphere_pars = [100, -80, 40, -7, -17, -7, None, None, None, None]

xyzuvw_perf_file = "perf_xyzuvw.npy"
result_file = "result.npy"
group_savefile = 'origins.npy'
#xyzuvw_init_savefile = 'xyzuvw_init.npy'
xyzuvw_init_datafile = 'data/sink_init_xyzuvw.npy'
astro_savefile = 'astro_table.txt'
xyzuvw_conv_savefile = 'xyzuvw_now.fits'
prec_val = {'perf': 1e-5, 'half':0.5, 'gaia': 1.0, 'double': 2.0}

BURNIN_STEPS = 1000
C_TOL = 0.15

print("In preamble")

# stops plots popping up as they are created, mayhaps too late if only
# put here....
try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not imported")
    pass

try:
    #age, dX, dV = np.array(sys.argv[1:4], dtype=np.double)
    age = float(sys.argv[1])
    precs = sys.argv[2:-1]
    package_path = sys.argv[-1]
except ValueError:
    print("Usage: ./perform_fed_fit.py [age] [prec1]"
          "[prec2] ... /relative/path/to/chronostar/")
    raise

# since this could be being executed anywhere, need to pass in package location
sys.path.insert(0, package_path)
try:
    import chronostar.synthesiser as syn
    import chronostar.traceorbit as torb
    import chronostar.converter as cv
    import chronostar.measurer as ms
    import chronostar.groupfitter as gf
    import chronostar.hexplotter as hp

except ImportError:
    #logging.info("Failed to import chronostar package")
    raise

# Initialize the MPI-based pool used for parallelization.
using_mpi = True
mpi_msg = ""    # can't use loggings yet, unclear if appending or rewriting
try:
    pool = MPIPool()
    mpi_msg += "Successfully initialised mpi pool"
except:
    #print("MPI doesn't seem to be installed... maybe install it?")
    mpi_msg += "MPI doesn't seem to be installed... maybe install it?"
    using_mpi = False
    pool=None

if using_mpi:
    if not pool.is_master():
        print("One thread is going to sleep")
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)
print("Only one thread is master")

# Setting up perfect current xyzuvw values
try:
    xyzuvw_now_perf = np.load(xyzuvw_perf_file)
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
    xyzuvw_init = np.load(package_path + '/' + xyzuvw_init_datafile)

    # incorporating offset so the group is measured at a consistent distance
    approx_final_pos = np.array([80, 100, -20])

    dist_travelled = 50. #pc
    vel_offset = np.array([dist_travelled/age, dist_travelled/age, 0.])
    offset = np.append([0,0,0], vel_offset)
    offset[0] -= age * offset[3]
    offset[1] -= age * offset[4]
    offset[2] -= age * offset[5]

    xyzuvw_init += offset
    xyzuvw_init[:,:3] += approx_final_pos
    xyzuvw_now_perf =\
        torb.traceManyOrbitXYZUVW(xyzuvw_init, age, single_age=True,
                                  savefile=xyzuvw_perf_file)

logging.info(mpi_msg)
if not using_mpi:
    logging.info("MPI available! - call this with e.g. mpirun -np 19"
                 " python perform_synth_fit.py")

# Performing fit for each precision
for prec in precs:
    logging.info("Fitting to prec: {}".format(prec))
    mkpath(prec)
    os.chdir(prec)
    #np.save(group_savefile, origin) # store in each directory, for hexplotter
    try:
        res = np.load(result_file)
        logging.info("Precision [{}] already fitted for".format(prec))
    except IOError:
        # convert XYZUVW data into astrometry
        astro_table = ms.measureXYZUVW(xyzuvw_now_perf, prec_val[prec],
                                       savefile=astro_savefile)
        star_pars =\
            cv.convertMeasurementsToCartesian(astro_table,
                                              savefile=xyzuvw_conv_savefile)
        logging.info("Generated [{}] traceback file".format(prec))

        # apply traceforward fitting (with lnprob, corner plots as side effects)
        best_fit, chain, lnprob = gf.fitGroup(
            xyzuvw_dict=star_pars, burnin_steps=BURNIN_STEPS, plot_it=True,
            pool=pool, convergence_tol=C_TOL
        )
        #hp.dataGatherer(save_dir=prec)

    finally:
        os.chdir('..')

if using_mpi:
    pool.close()
