#! /usr/bin/env python -W ignore
"""
This script demos the use of tfgroupfitter. It determines the most likely
origin point of a set of stars assuming a (separate) spherical distribution in
position and velocity space.

Call with:
python perform_tf_fit.py [age] [dX] [dV] [nstars] [prec..] [path_to_chronostar]
or
mpirun -np [nthreads] python perform_tf_fit.py [age] [dX] [dV] [nstars] [prec..]
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
import pickle
import sys
from emcee.utils import MPIPool

init_stars_file = "init_stars.npy"
perf_data_file = "perf_xyzuvw.npy"
result_file = "result.npy"
prec_val = {'perf': 1e-5, 'half':0.5, 'gaia': 1.0, 'double': 2.0}


BURNIN_STEPS = 1000
C_TOL = 0.2

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
    data_path = package_path + "/data/sink_init_xyzuvw.npy"
except (IndexError, NameError, ValueError):
    print("Usage: ./perform_tf_fit.py [age] [prec1]"
          "[prec2] ... /relative/path/to/chronostar/")
    raise

# since this could be being executed anywhere, need to pass in package location
sys.path.insert(0, package_path)
try:
    import chronostar.retired.synthesiser as syn
    import chronostar.retired.tracingback as tb
    import chronostar.retired.tfgroupfitter as tfgf
    import chronostar.retired.error_ellipse as ee
    import chronostar.transform as tf
    from chronostar.retired import utils
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

try:
    init_xyzuvw = np.load(init_stars_file)
except IOError:
    ## decrement position by approx vel*t so final result is
    ## in similar location across ages

    # approx region of present day positions
    approx_final_pos = np.array([80, 100, -20])
    # velocities set that group travels approx 50 pc through X-Y plane
    dist_travelled = 50. #pc
    vel_offset = np.array([dist_travelled/age, dist_travelled/age, 0.])
    offset = np.append([0,0,0], vel_offset)
    offset[0] += age * offset[3]
    offset[1] -= age * offset[4]
    offset[2] -= age * offset[5]

    init_xyzuvw = np.load(data_path)
    init_xyzuvw += offset
    init_xyzuvw[:,:3] += approx_final_pos
    np.save(init_stars_file, init_xyzuvw)

try:
    perf_xyzuvws = np.load(perf_data_file)
    logging.basicConfig(
        level=logging.DEBUG, filemode='a',
        filename='my_investigator_demo.log',
    )
    logging.info("appending to previous attempt")
except IOError:
    logging.basicConfig(
        level=logging.DEBUG, filemode='w',
        filename='my_investigator_demo.log',
    )
    logging.info("Beginning fresh run:")
    logging.info("Input arguments: {}".format(sys.argv[1:]))
    logging.info("\n"
                 "\tage:     {}\n"
#                 "\tdX:      {}\n"
#                 "\tdV:      {}\n"
#                 "\tnstars:  {}\n"
                 "\tprecs:   {}".format(
        age, precs,
    ))

    # synthesise perfect XYZUVW data
    logging.info("Synthesising data")
#    perf_xyzuvws, _ = syn.generate_current_pos(0, None, )
#    np.save(perf_data_file, perf_xyzuvws)
    syn.synthesise_data(None, None, init_xyzuvw=init_xyzuvw, age=age,
                        perf_data_file=perf_data_file)
    perf_xyzuvws = np.load(perf_data_file)

logging.info(mpi_msg)
if not using_mpi:
    logging.info("MPI available! - call this with e.g. mpirun -np 19"
                 " python perform_tf_fit.py")

# calculating all the relevant covariance matrices
# init_xyzuvw = np.load(init_xyzuvw)
then_cov_true = np.cov(init_xyzuvw.T)

dXav = (np.prod(np.linalg.eigvals(then_cov_true[:3, :3])) ** (1. / 6.))

for prec in precs:
    # if we are being PEDANTIC can also check if traceback
    # measurements have already been made, and skip those
    # but honestly, too much noise atm

    logging.info("Fitting to prec: {}".format(prec))
    mkpath(prec)
    os.chdir(prec)

    # check if this precision has already been fitted
    try:
        res = np.load(result_file)
        logging.info("Precision [{}] already fitted for".format(prec))
    except IOError:
        # convert XYZUVW data into astrometry
        sky_coord_now = syn.measure_stars(perf_xyzuvws)
        synth_table = syn.generate_table_with_error(
            sky_coord_now, prec_val[prec]
        )
        astr_file = "astr_data"
        pickle.dump(synth_table, open(astr_file, 'w'))

        # convert astrometry back into XYZUVW data
        tb_file = "tb_data.pkl"
        tb.traceback(synth_table, np.array([0,1]), savefile=tb_file)
        logging.info("Generated [{}] traceback file".format(prec))

        # apply traceforward fitting (with lnprob, corner plots as side effects)
        best_fit, chain, lnprob = tfgf.fit_group(
            tb_file, burnin_steps=BURNIN_STEPS, plot_it=True, pool=pool,
            convergence_tol=C_TOL,
        )


        # plot Hex plot TODO, atm, just got a simple res plot going
        #star_pars = tfgf.read_stars(tb_file=tb_file)
#            xyzuvw = star_pars['xyzuvw'][:,0]
#            xyzuvw_cov = star_pars['xyzuvw_cov'][:,0]

#        means = {}
#        covs  = {}

#        # save and store result so hex-plots can be calculated after the fact
#        means['origin_then'] = np.array([group_pars_ex[:6]])
#        np.save(result_file, [best_fit, chain, lnprob, group_pars_in, group_pars_tf_style, group_pars_ex])
#
#        #then_cov_true
#        covs['origin_then'] = np.array([
#            utils.generate_cov( utils.internalise_pars(group_pars_ex))
#        ])
#
#        then_cov_simple = tfgf.generate_cov(group_pars_in)
#        means['fitted_then'] = np.array([
#            best_fit[0:6]
#        ])
#        covs['fitted_then'] = np.array([
#            tfgf.generate_cov(best_fit)
#        ])
#
#        means['fitted_now'] = np.array([
#            tb.trace_forward(best_fit[:6], best_fit[-1])
#        ])
#        covs['fitted_now'] = np.array([
#            tf.transform_cov(covs['fitted_then'][0], tb.trace_forward,
#                             means['fitted_then'][0], dim=6,
#                             args=(best_fit[-1],))
#        ])
#        np.save('covs.npy', covs)
#        np.save('means.npy', means)

        #plt.plot(xyzuvw[:, 0], xyzuvw[:, 1], 'b.')
        #ee.plot_cov_ellipse(then_cov_simple[:2, :2], group_pars_tf_style[:2],
        #                    color='orange',
        #                    alpha=0.2, hatch='|', ls='--')
        #ee.plot_cov_ellipse(then_cov_true[:2, :2], group_pars_tf_style[:2],
        #                    color='orange',
        #                    alpha=1, ls=':', fill=False)
        #ee.plot_cov_ellipse(then_cov_fitted[:2, :2], best_fit[:2],
        #                    color='xkcd:neon purple',
        #                    alpha=0.2, hatch='/', ls='-.')
        #ee.plot_cov_ellipse(now_cov_fitted[:2, :2], now_mean_fitted[:2],
        #                    color='b',
        #                    alpha=0.03, hatch='.')

        #buffer = 30
        #xmin = min(group_pars_tf_style[0], best_fit[0], now_mean_fitted[0], *xyzuvw[:,0])
        #xmax = max(group_pars_tf_style[0], best_fit[0], now_mean_fitted[0], *xyzuvw[:,0])
        #ymin = min(group_pars_tf_style[1], best_fit[1], now_mean_fitted[1], *xyzuvw[:,1])
        #ymax = max(group_pars_tf_style[1], best_fit[1], now_mean_fitted[1], *xyzuvw[:,1])
        #plt.xlim(xmax + buffer, xmin - buffer)
        #plt.ylim(ymin - buffer, ymax + buffer)
        #plt.title("age: {}, dX: {}, dV: {}, nstars: {}, prec: {}".format(
        #    age, dX, dV, nstars, prec
        #))
        #plt.savefig("XY_plot_{}_{}_{}_{}_{}.png".format(
        #    age, dX, dV, nstars, prec
        #))

        #plt.clf()
        #plt.hist(chain[:,:,-1].flatten(), bins=20)
        #plt.title("age: {}, dX: {}, dV: {}, nstars: {}, prec: {}".format(
        #    age, dX, dV, nstars, prec
        #))
        #plt.savefig("age_hist_{}_{}_{}_{}_{}.png".format(
        #    age, dX, dV, nstars, prec
        #))

        # return to main directory
    finally:
        os.chdir('..')

if using_mpi:
    pool.close()
