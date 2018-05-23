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

try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits


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

    err_arr_cp[:, :2] = 1e-1
    logging.info("Angular position error is: {}".format(err_arr_cp[0,0]))
    print("Angular position error is: {}".format(err_arr_cp[0,0]))
    print("Changed!")

    astr_covs = np.zeros((nstars, 6, 6))
    for ix in range(nstars):
        astr_covs[ix] = np.eye(6) * np.tile(err_arr_cp[ix], (6, 1))**2
    return astr_covs


def transformAstrCovsToCartesian(astr_covs, astr_arr):
    """
    Converts a covariance matrix from astrometric coords to LSR XYZUVW
    Parameters
    ----------
    astr_covs: ([nstars, 6, 6] array)
        values in the diagaonal are the squared errors of
        (ra, dec, plx, pm_ra, pm_dec, rv), with the offdiagonals the product
        of the correlation (valued between -1 and 1) and the two
        intersecting coordinates.
    astr_arr: ([nstars, 6] array)
        the measured (mean) astrometric values
        (ra, dec, plx, pm_ra, pm-dec, rv)
    """
    nstars = astr_arr.shape[0]
    xyzuvw_covs = np.zeros((nstars, 6, 6))
    for ix in range(nstars):
        xyzuvw_covs[ix] = tf.transform_cov(
            astr_covs[ix], cc.convertAstrometryToLSRXYZUVW, astr_arr[ix],
            dim=6
        )
    return xyzuvw_covs


def saveDictAsFits(savefile, xyzuvw_dict):
    """Convert dict into fits format and save"""
    if (savefile[-3:] != 'fit') and (savefile[-4:] != 'fits'):
        savefile = savefile + ".fits"
    hl = pyfits.HDUList()
    hl.append(pyfits.PrimaryHDU())
    hl.append(pyfits.ImageHDU(xyzuvw_dict['xyzuvw']))
    hl.append(pyfits.ImageHDU(xyzuvw_dict['xyzuvw_cov']))
    # TODO: Get Mike to help with this step
    #hl.append(pyfits.TableHDU(xyzuvw_dict['table'])) #
    hl.writeto(savefile, overwrite=True)


def convertMeasurementsToCartesian(t=None, loadfile='', astr_dict=None,
                                   savefile=''):
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
    loadfile : (String {''})
        if t is None, try and load table from loadfile
    savefile : (String {''})
        if non-empty, will save a fits file with this filename. Appropriate
        file extension is applied if not there.

    Returns
    -------
    dict :
        t : astropy table
        xyzuvw : [nstars, 6] array
            space positions and velocities of each star
        xyzuvw_cov : [nstars, 6, 6] array
            covariance of positions and velocities of each star
    """
    while True:
        if t:
            nstars = len(t)
            astr_arr, err_arr = ms.convertTableToArray(t)
            astr_covs = convertAstrErrsToCovs(err_arr)
            break
        if loadfile:
            t = Table.read(loadfile, format='ascii')
            nstars = len(t)
            astr_arr, err_arr = ms.convertTableToArray(t)
            astr_covs = convertAstrErrsToCovs(err_arr)
            break
        if astr_dict:
            astr_arr = astr_dict['astr_mns']
            astr_covs = astr_dict['astr_covs']
            nstars = astr_arr.shape[0]
            break
        raise StandardError


#    nstars = len(t)
#    astr_arr, err_arr = ms.convertTableToArray(t)
#    astr_covs = convertAstrErrsToCovs(err_arr)

    xyzuvw = cc.convertManyAstrometryToLSRXYZUVW(astr_arr, mas=True)
    xyzuvw_cov = transformAstrCovsToCartesian(astr_covs, astr_arr)

    xyzuvw_dict = {'table':t, 'xyzuvw':xyzuvw, 'xyzuvw_cov':xyzuvw_cov}

    if savefile:
        saveDictAsFits(savefile, xyzuvw_dict)

    return xyzuvw_dict
