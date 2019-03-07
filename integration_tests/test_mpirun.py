from emcee.utils import MPIPool
import logging
import numpy as np
import sys

import chronostar.component
import chronostar.synthdata

sys.path.insert(0,'..')

import chronostar.groupfitter as gf
import chronostar.retired2.converter as cv
import chronostar.traceorbit as torb
import chronostar.synthdata as syn

def test_mpirun():
    logging.basicConfig(level=logging.INFO, filename='logs/mpirun.log')
    try:
        pool = MPIPool()
        using_mpi = True
    except:
        logging.error("Need to have mpi installed; need to run with:\n"
                      "mpirun -np 19 python test_mpirun.py")
        using_mpi = False
        pool = None
        raise

    logging.info("Using_mpi: {}".format(using_mpi))
    if using_mpi:
        if not pool.is_master():
            logging.info("This thread is waiting...")
            pool.wait()
            sys.exit(0)

    logging.info("This thread is the master!")

    save_dir = 'temp_data/'
    group_savefile = save_dir + 'origins.npy'
    xyzuvw_init_savefile = save_dir + 'xyzuvw_init.npy'
    astro_savefile = save_dir + 'astro_table.txt'
    xyzuvw_conv_savefile = save_dir + 'xyzuvw_now.fits'

    pars = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1e-8, 100])
    error_frac = 1.0
    xyzuvw_init, group = syn.synthesiseXYZUVW(pars, return_group=True,
                                              xyzuvw_savefile=xyzuvw_init_savefile,
                                              group_savefile=group_savefile,
                                              internal=True)
    xyzuvw_now = torb.traceManyOrbitXYZUVW(xyzuvw_init, group.age,
                                           single_age=True)
    astro_table = chronostar.synthdata.measureXYZUVW(xyzuvw_now, error_frac,
                                                     savefile=astro_savefile)
    star_pars = cv.convertMeasurementsToCartesian(astro_table,
                                                  savefile=xyzuvw_conv_savefile)

    best_fit, chain, lnprob = gf.fit_group(
        xyzuvw_dict=star_pars, plot_it=True, convergence_tol=0.4,
        burnin_steps=1000, pool=pool, plot_dir='temp_plots/',
        save_dir='temp_data/'
    )

    best_fit_group = chronostar.component.Component(best_fit, internal=True)

    assert np.allclose(best_fit_group.mean, group.mean, atol=0.5)
    assert np.allclose(best_fit_group.age, group.age, atol=0.5)
    assert np.allclose(best_fit_group.generateCovMatrix(),
                       group.generateCovMatrix(), atol=0.5)

if __name__ == '__main__':
    test_mpirun()
