import logging
import numpy as np
import sys

sys.path.insert(0,'..')

import chronostar.groupfitter as gf
import chronostar.converter as cv
import chronostar.measurer as ms
import chronostar.synthesiser as syn

def test_stationaryGroup():
    logging.basicConfig(level=logging.INFO, filename='logs/groupfitter.log')
    save_dir = 'temp_data/'
    group_savefile = save_dir + 'origins_stat.npy'
    xyzuvw_init_savefile = save_dir + 'xyzuvw_init_stat.npy'
    astro_savefile = save_dir + 'astro_table_stat.txt'
    xyzuvw_conv_savefile = save_dir + 'xyzuvw_conv_stat.fits'

    pars = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1e-8, 100])
    error_frac = 1.0
    xyzuvw_init, group = syn.synthesise_xyzuvw(
        pars, return_group=True, internal=True, group_savefile=group_savefile,
        xyzuvw_savefile=xyzuvw_init_savefile
    )
    astro_table = ms.measureXYZUVW(xyzuvw_init, error_frac,
                                   savefile=astro_savefile)
    star_pars = cv.convertMeasurementsToCartesian(astro_table,
                                                  savefile=xyzuvw_conv_savefile)

    best_fit, chain, lnprob = gf.fitGroup(xyzuvw_dict=star_pars, plot_it=True,
                                          convergence_tol=0.4, burnin_steps=300,
                                          plot_dir='temp_plots/',
                                          save_dir='temp_data/')
    best_fit_group = syn.Group(best_fit, internal=True)

    assert np.allclose(best_fit_group.mean, group.mean, atol=0.5)
    assert np.allclose(best_fit_group.age, group.age, atol=0.5)
    assert np.allclose(best_fit_group.generateCovMatrix(),
                       group.generateCovMatrix(), atol=0.5)

if __name__ == '__main__':
    test_stationaryGroup()
