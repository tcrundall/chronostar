import logging
import numpy as np
from astropy.io import fits
import sys

sys.path.insert(0, '..')

import chronostar.converter as cv
import chronostar.coordinate as cc
import chronostar.synthesiser as syn
import chronostar.transform as tf
import chronostar.traceorbit as to
import chronostar.measurer as ms

def test_converter():
    """Fairly in depth integration test"""
    AGE = 1e-5

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    save_dir = 'temp_data/'
    group_savefile = save_dir + 'origins.npy'
    xyzuvw_init_savefile = save_dir + 'xyzuvw_init.npy'
    astro_savefile = save_dir + 'astro_table.txt'
    xyzuvw_conv_savefile = save_dir + 'xyzuvw_conv.fits'

    group_pars = [0., 0., 0., 0., 0., 0., 1., 1., AGE, 100]
    xyzuvw_init, group = syn.synthesiseXYZUVW(
        group_pars, sphere=True, xyzuvw_savefile=xyzuvw_init_savefile,
        group_savefile=group_savefile, return_group=True
    )
    xyzuvw_now_true = to.traceManyOrbitXYZUVW(xyzuvw_init, group.age,
                                              single_age=True)

    ms.measureXYZUVW(xyzuvw_now_true, 1.0, astro_savefile)
    cv.convertMeasurementsToCartesian(loadfile=astro_savefile,
                                      savefile=xyzuvw_conv_savefile)

    xyzuvw_now = fits.getdata(xyzuvw_conv_savefile, 1) #hdulist[1].data
    xyzuvw_cov_now = fits.getdata(xyzuvw_conv_savefile, 2) #hdulist[2].data

    assert np.allclose(np.mean(xyzuvw_now, axis=0), group.mean, atol=0.5)

    logging.info("Approx cov:\n{}".format(np.cov(xyzuvw_now.T)))
    logging.info("Comparing with:\n{}".format(
        group.generateSphericalCovMatrix()
    ))
    assert np.allclose(np.cov(xyzuvw_now.T), group.generateSphericalCovMatrix(),
                       atol=0.5)

