import logging
import numpy as np
import sys

import chronostar.component
import chronostar.synthdata

sys.path.insert(0,'..')

import chronostar.groupfitter as gf
import chronostar.converter as cv
import chronostar.measurer as ms
import chronostar.synthdata as syn

def test_stationaryGroup():
    """
    Integrated test which fits a single component to a synthetic association.

    Runtime on my mac (single thread) is ~ 20 mins. Check logs/groupfitter.log
    and temp_plots/*.png for progress.
    """
    logging.basicConfig(level=logging.INFO, filename='logs/groupfitter.log',
                        filemode='w')
    save_dir = 'temp_data/'
    group_savefile = save_dir + 'origins_stat.npy'
    xyzuvw_init_savefile = save_dir + 'xyzuvw_init_stat.npy'
    astro_savefile = save_dir + 'astro_table_stat.txt'
    xyzuvw_conv_savefile = save_dir + 'xyzuvw_conv_stat.fits'

    pars = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1e-8, 100])
    error_frac = 0.1
    xyzuvw_init, group = syn.synthesiseXYZUVW(pars, return_group=True,
                                              xyzuvw_savefile=xyzuvw_init_savefile,
                                              group_savefile=group_savefile,
                                              internal=True)
    astro_table = chronostar.synthdata.measureXYZUVW(xyzuvw_init, error_frac,
                                                     savefile=astro_savefile)
    star_pars = cv.convertMeasurementsToCartesian(astro_table,
                                                  savefile=xyzuvw_conv_savefile)

    best_fit, chain, lnprob = gf.fitGroup(
        xyzuvw_dict=star_pars, plot_it=True, convergence_tol=0.25,
        burnin_steps=400, plot_dir='temp_plots/', save_dir='temp_data/'
    )
    best_fit_group = chronostar.component.Component(best_fit, internal=True)

    assert np.allclose(best_fit_group.mean, group.mean, atol=0.5)
    assert np.allclose(best_fit_group.age, group.age, atol=0.5)
    assert np.allclose(best_fit_group.generateCovMatrix(),
                       group.generateCovMatrix(), atol=0.5)

if __name__ == '__main__':
    test_stationaryGroup()
