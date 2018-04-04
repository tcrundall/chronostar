#! /usr/bin/env python
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


base_group_pars = [
    -80, 80, 50, 10, -20, -5, None, None, None, None,
    0.0, 0.0, 0.0, None, None
]
perf_data_file = "perf_xyzuvw.npy"
result_file = "result.npy"
prec_val = {'perf': 1e-5, 'half':0.5, 'gaia': 1.0, 'double': 2.0}


BURNIN_STEPS = 500
if __name__ == '__main__':

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
            # Wait for instructions from the master process.
            pool.wait()
            sys.exit(0)

    try:
        age, dX, dV = np.array(sys.argv[1:4], dtype=np.double)
        nstars = int(sys.argv[4])
        precs = sys.argv[5:-1]
        package_path = sys.argv[-1]
    except ValueError:
        print("Usage: ./perform_tf_fit.py [age] [dX] [dV] [nstars] [prec1]"
              "[prec2] ... /relative/path/to/chronostar/")
        raise

    # since this could be being executed anywhere, need to pass in package location
    sys.path.insert(0, package_path)
    try:
        import chronostar.synthesiser as syn
        import chronostar.traceback as tb
        import chronostar.tfgroupfitter as tfgf
        import chronostar.error_ellipse as ee
        import chronostar.transform as tf
        from chronostar import utils
    except ImportError:
        #logging.info("Failed to import chronostar package")
        raise

    # collect inputs
    group_pars_ex = list(base_group_pars)
    group_pars_ex[6:9] = [dX, dX, dX]
    group_pars_ex[9] = dV
    group_pars_ex[13] = age
    group_pars_ex[14] = nstars

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
                     "\tdX:      {}\n"
                     "\tdV:      {}\n"
                     "\tnstars:  {}\n"
                     "\tprecs:   {}".format(
            age, dX, dV, nstars, precs,
        ))

        # synthesise perfect XYZUVW data
        logging.info("Synthesising data")
        perf_xyzuvws, _ = syn.generate_current_pos(1, group_pars_ex)
        np.save(perf_data_file, perf_xyzuvws)

    logging.info(mpi_msg)
    if not using_mpi:
        logging.info("MPI available! - call this with e.g. mpirun -np 4"
                     " python fitting_TWA.py")

    for prec in precs:
        # if we are being PEDANTIC can also check if traceback
        # measurements have already been made, and skip those
        # but honestly, too much noise atm        

        logging.info("Fitting to prec: {}".format(prec))
        mkpath(prec)
        os.chdir(prec)
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

            # apply traceforward fitting (with lnprob, corner plots as side effects)
            best_fit, chain, lnprob = tfgf.fit_group(
                tb_file, burnin_steps=BURNIN_STEPS, plot_it=True, pool=pool,
            )

            # plot Hex plot TODO, atm, just got a simple res plot going
            star_pars = tfgf.read_stars(tb_file=tb_file)
            xyzuvw = star_pars['xyzuvw'][:,0]
            xyzuvw_cov = star_pars['xyzuvw_cov'][:,0]

            # calculating all the relevant covariance matrices
            then_cov_true = utils.generate_cov(utils.internalise_pars(
                group_pars_ex
            ))

            dXav = (np.prod(np.linalg.eigvals(then_cov_true[:3, :3])) ** (1. / 6.))

            # This represents the target result - a simplified, spherical
            # starting point
            group_pars_tf_style = \
                np.append(
                    np.append(
                        np.append(np.copy(group_pars_ex)[:6], dXav), dV
                    ), age
                )
            group_pars_in = np.copy(group_pars_tf_style)
            group_pars_in[6:8] = np.log(group_pars_in[6:8])

            # save and store result so hex-plots can be calculated after the fact
            np.save(result_file, [best_fit, chain, lnprob, group_pars_ex, group_pars_tf_style])

            then_cov_true = utils.generate_cov(
                utils.internalise_pars(group_pars_ex))
            then_cov_simple = tfgf.generate_cov(group_pars_in)
            then_cov_fitted = tfgf.generate_cov(best_fit)
            now_cov_fitted = tf.transform_cov(then_cov_fitted, tb.trace_forward,
                                              best_fit[0:6], dim=6,
                                              args=(best_fit[-1],))
            now_mean_fitted = tb.trace_forward(best_fit[:6], best_fit[-1])

            plt.clf()
            plt.plot(xyzuvw[:, 0], xyzuvw[:, 1], 'b.')
            ee.plot_cov_ellipse(then_cov_simple[:2, :2], group_pars_tf_style[:2],
                                color='orange',
                                alpha=0.2, hatch='|', ls='--')
            ee.plot_cov_ellipse(then_cov_true[:2, :2], group_pars_tf_style[:2],
                                color='orange',
                                alpha=1, ls=':', fill=False)
            ee.plot_cov_ellipse(then_cov_fitted[:2, :2], best_fit[:2],
                                color='xkcd:neon purple',
                                alpha=0.2, hatch='/', ls='-.')
            ee.plot_cov_ellipse(now_cov_fitted[:2, :2], now_mean_fitted[:2],
                                color='b',
                                alpha=0.03, hatch='.')

            buffer = 30
            xmin = min(group_pars_tf_style[0], best_fit[0], now_mean_fitted[0], *xyzuvw[:,0])
            xmax = max(group_pars_tf_style[0], best_fit[0], now_mean_fitted[0], *xyzuvw[:,0])
            ymin = min(group_pars_tf_style[1], best_fit[1], now_mean_fitted[1], *xyzuvw[:,1])
            ymax = max(group_pars_tf_style[1], best_fit[1], now_mean_fitted[1], *xyzuvw[:,1])
            plt.xlim(xmax + buffer, xmin - buffer)
            plt.ylim(ymin - buffer, ymax + buffer)
            plt.title("age: {}, dX: {}, dV: {}, nstars: {}, prec: {}".format(
                age, dX, dV, nstars, prec
            ))
            plt.savefig("XY_plot_{}_{}_{}_{}_{}.png".format(
                age, dX, dV, nstars, prec
            ))

            plt.clf()
            plt.hist(chain[:,:,-1].flatten(), bins=20)
            plt.title("age: {}, dX: {}, dV: {}, nstars: {}, prec: {}".format(
                age, dX, dV, nstars, prec
            ))
            plt.savefig("age_hist_{}_{}_{}_{}_{}.png".format(
                age, dX, dV, nstars, prec
            ))

            # return to main directory
        finally:
            os.chdir('..')

