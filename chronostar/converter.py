"""
Module which converts astrometry measurements and errors into equivalent
mean and covariance matrix in XYZUVW space
"""

from astropy.table import Table
import logging
import numpy as np

import measurer as ms
import coordinate as cc
import transform as tf

def convertAstrErrsToCovs(err_arr):
    """
    Converts astrometry errors for each star into covariance matrices

    Note that a negligible error is inserted for ra and dec

    Parameters
    ----------
    err_arr : ([nstars, 6] float array), astrometry errors with placeholder
        0's for ra and dec: (ra, dec, pi, pm_ra, pm_dec, rv)

    Returns
    -------
    astr_covs : ([nstars, 6, 6] float array), covariance matrix made by
        inserting the square of the errors into the diagonal
    """
    err_arr_cp = np.copy(err_arr)
    nstars = err_arr_cp.shape[0]

    err_arr_cp[:, :2] = 5e-2
    logging.info("Angular position error is: {}".format(err_arr_cp[0,0]))
    print("Angular position error is: {}".format(err_arr_cp[0,0]))

    astr_covs = np.zeros((nstars, 6, 6))
    for ix in range(nstars):
        astr_covs[ix] = np.eye(6) * np.tile(err_arr_cp[ix], (6, 1))**2
    return astr_covs

def transformAstrCovsToCartesian(astr_covs, astr_arr):
    nstars = astr_arr.shape[0]
    xyzuvw_covs = np.zeros((nstars, 6, 6))
    for ix in range(nstars):
        xyzuvw_covs[ix] = tf.transform_cov(
            astr_covs[ix], cc.convertAstrmetryToLSRXYZUVW, astr_arr[ix], dim=6
        )
    return xyzuvw_covs


def convertMeasurementsToCartesian(t=None, loadfile='', savefile=''):
    """
    Parameters
    ----------
    t : astropy Table with the following columns:
        name    : id or name of star
        radeg   : right ascension in degrees
        dedeg   : declination in degrees
        plx     : parallax in mas
        e_plx   : error of parallax in mas
        pmra    : proper motion in right ascension in mas/yr
        e_pmra  : error of pm in right ascension in mas/yr
        pmde    : proper motion in declination in mas/yr
        e_pmde  : error of pm in declination in mas/yr
        rv      : radial velocity in km/s
        e_rv    : error of radial velocity in km/s

    Returns
    -------
    dict :
        t : astropy table
        xyzuvw : [nstars, 6] array
            space positions and velocities of each star
        xyzuvw_cov : [nstars, 6, 6] array
            covariance of positions and velocities of each star
    """
    if t is None:
        t = Table.read(loadfile, format='ascii')

    nstars = len(t)
    astr_arr, err_arr = ms.convertTableToArray(t)
    astr_covs = convertAstrErrsToCovs(err_arr)

    xyzuvw = cc.convertManyAstrometryToLSRXYZUVW(astr_arr, mas=True)
    xyzuvw_cov = transformAstrCovsToCartesian(astr_covs, astr_arr)

    return {'table':t, 'xyzuvw':xyzuvw, 'xyzuvw_cov':xyzuvw_cov}
