import logging
import numpy as np
from astropy.io import fits
import pdb
import sys

import chronostar.synthdata

sys.path.insert(0, '..')

import chronostar.converter as cv
import chronostar.coordinate as cc
import chronostar.synthdata as syn
import chronostar.transform as tf
import chronostar.traceorbit as to
import chronostar.measurer as ms

def test_converter():
    """Fairly in depth integration test

    Synthesises an association in XYZUVW
    at LSR (i.e. -25 pc from the sun in Z
    direction), traces forward a negligible time step, converts to
    astrometric measurement with some small error, converts back
    to XYZUVW and compares distribution to initialising covariance
    matrix.

    Note, if measurement uncertainty is allowed to be large (>0.7?)
    then points are scattered in W so much that the covariance matrix
    comparision fails.

    This is because the largest contributer to uncertainty is radial
    velocity, and since the association is initialised at (0,0,0) and
    measured from the sun at (0,0,25), the uncertainty translates to
    an uncertainty in W (the vertical component of space velocity).
    """
    AGE = 1e-5

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    save_dir = 'temp_data/'
    group_savefile = save_dir + 'origins.npy'
    xyzuvw_init_savefile = save_dir + 'xyzuvw_init.npy'
    astro_savefile = save_dir + 'astro_table.txt'
    xyzuvw_conv_savefile = save_dir + 'xyzuvw_conv.fits'

    # Generate 100 synthetic stars centred at LSR with dX = dV = 1
    group_pars = [0., 0., 0., 0., 0., 0., 1., 1., AGE, 100]
    xyzuvw_init, group = syn.synthesiseXYZUVW(group_pars, form='sphere',
                                              return_group=True,
                                              xyzuvw_savefile=xyzuvw_init_savefile,
                                              group_savefile=group_savefile)

    # Traceforward by a negligible time step
    xyzuvw_now_true = to.traceManyOrbitXYZUVW(xyzuvw_init, group.age,
                                              single_age=True)

    # Measure with Gaia-ish uncertainty
    # ms.measureXYZUVW(xyzuvw_now_true, 1.0, astro_savefile)
    chronostar.synthdata.measureXYZUVW(xyzuvw_now_true, 0.1, astro_savefile)
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

