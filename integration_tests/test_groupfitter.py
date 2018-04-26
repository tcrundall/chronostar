import logging
import numpy as np
import sys

sys.path.insert(0,'..')

import chronostar.groupfitter as gf
import chronostar.converter as cv
import chronostar.measurer as ms
import chronostar.synthesiser as syn
import chronostar.traceorbit as torb

def test_stationaryGroup():
    logging.basicConfig(level=logging.INFO, filename='logs/groupfitter.log')
    #save_dir = 'temp_data/'
    #group_savefile = save_dir + 'origins.npy'
    #xyzuvw_init_savefile = save_dir + 'xyzuvw_init.npy'
    #astro_savefile = save_dir + 'astro_table.txt'
    #xyzuvw_conv_savefile = save_dir + 'xyzuvw_conv.fits'

    pars = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1e-8, 100])
    error_frac = 1.0
    xyzuvw_init, group = syn.synthesise_xyzuvw(pars, return_group=True,
                                               internal=True)
    astro_table = ms.measureXYZUVW(xyzuvw_init, error_frac)
    star_pars = cv.convertMeasurementsToCartesian(astro_table)

    best_fit, chain, lnprob = gf.fitGroup(xyzuvw_dict=star_pars)
    best_fit_group = syn.Group(best_fit, internal=True)

    assert np.allclose(best_fit_group.mean, group.mean, atol=0.5)
    assert np.allclose(best_fit_group.age, group.age, atol=0.5)
    assert np.allclose(best_fit_group.generateCovMatrix(),
                       group.generateCovMatrix(), atol=0.5)
