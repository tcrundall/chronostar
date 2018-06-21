"""
A suite of useful functions for partitioning the gaia data
TODO: write tests especially for cov construction
"""

from astropy.io import fits
import numpy as np
import logging
import sys
sys.path.insert(0, '..')

import chronostar.converter as cv
import chronostar.coordinate as cc

def createSubFitsFile(mask, filename):
    """
    Provide a mask (constructed based on Gaia DR2 fits) to build new one

    Parameters
    ----------
    mask : [nstars] int array in tuple
        The output of np.where applying some filter to gaia data
        e.g.
            np.where(hdul[1].data[:,1] > 0)
        produces a mask to grab all stars with positive DEC
    filename : string
        name of destination fits file
    """
    if filename[-4] == "fits":
        filename += ".fits"
    gaia_file = "../data/all_rvs_w_ok_plx.fits"
    with fits.open(gaia_file) as hdul:
        primary_hdu = fits.PrimaryHDU(header=hdul[1].header)
        hdu = fits.BinTableHDU(data=hdul[1].data[mask])
        new_hdul = fits.HDUList([primary_hdu, hdu])
        new_hdul.writeto(filename)


def convertRecToArray(sr):
    """UNTESTED"""
    ra = sr['ra']
    e_ra = sr['ra_error'] / 3600. / 1000.
    dec = sr['dec']
    e_dec = sr['dec_error'] / 3600. / 1000.
    plx = sr['parallax']
    e_plx = sr['parallax_error']
    pmra = sr['pmra']
    e_pmra = sr['pmra_error']
    pmdec = sr['pmdec']
    e_pmdec = sr['pmdec_error']
    rv = sr['radial_velocity']
    e_rv = sr['radial_velocity_error']
    c_ra_dec = sr['ra_dec_corr']
    c_ra_plx = sr['ra_parallax_corr']
    c_ra_pmra = sr['ra_pmra_corr']
    c_ra_pmdec = sr['ra_pmdec_corr']
    c_dec_plx = sr['dec_parallax_corr']
    c_dec_pmra = sr['dec_pmra_corr']
    c_dec_pmdec = sr['dec_pmdec_corr']
    c_plx_pmra = sr['parallax_pmra_corr']
    c_plx_pmdec = sr['parallax_pmdec_corr']
    c_pmra_pmdec = sr['pmra_pmdec_corr']

    mean = np.array((ra, dec, plx, pmra, pmdec, rv))
    cov  = np.array([
        [e_ra**2, c_ra_dec*e_ra*e_dec, c_ra_plx*e_ra*e_plx,
            c_ra_pmra*e_ra*e_pmra, c_ra_pmdec*e_ra*e_pmdec, 0.],
        [c_ra_dec*e_ra*e_dec, e_dec**2, c_dec_plx*e_dec*e_plx,
            c_dec_pmra*e_dec*e_pmra, c_dec_pmdec*e_dec*e_pmdec, 0.],
        [c_ra_plx*e_ra*e_plx, c_dec_plx*e_dec*e_plx, e_plx**2,
            c_plx_pmra*e_plx*e_pmra, c_plx_pmdec*e_plx*e_pmdec, 0.],
        [c_ra_pmra*e_ra*e_pmra, c_dec_pmra*e_dec*e_pmra,
                                                c_plx_pmra*e_plx*e_pmra,
             e_pmra**2, c_pmra_pmdec*e_pmra*e_pmdec, 0.],
        [c_ra_pmdec*e_ra*e_pmdec, c_dec_pmdec*e_dec*e_pmdec,
                                                c_plx_pmdec*e_plx*e_pmdec,
             c_pmra_pmdec*e_pmra*e_pmdec, e_pmdec**2, 0.],
        [0., 0., 0., 0., 0., e_rv**2]
    ])
    return mean, cov

def convertManyRecToArray(data):
    """
    Convert many Fits Records into mean and covs

    Note: ra_error and dec_error are in 'mas' while ra and dec
    are given in degrees. Everything else is standard:
    plx [mas], pm [mas/yr], rv [km/s]

    Parameters
    ----------
    data: [nstars] array of Recs
        'source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax',
        'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
        'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr',
        'ra_pmdec_corr', 'dec_parallax_corr', 'dec_pmra_corr',
        'dec_pmdec_corr', 'parallax_pmra_corr', 'parallax_pmdec_corr',
        'pmra_pmdec_corr', 'astrometric_primary_flag', 'phot_g_mean_mag',
        'radial_velocity', 'radial_velocity_error', 'teff_val'

    Returns
    -------
        means: [nstars, 6] array
        covs: [nstars, 6] array
    """
    logging.info("In convertManyRecToArray")
    nstars = data.shape[0]
    print("nstars: {}".format(nstars))
    logging.info("nstars: {}".format(nstars))
    means = np.zeros((nstars,6))

    means[:,0] = data['ra']
    means[:,1] = data['dec']
    means[:,2] = data['parallax']
    means[:,3] = data['pmra']
    means[:,4] = data['pmdec']
    means[:,5] = data['radial_velocity']

    # Array of dictionary keys to aid construction of cov matrices
    cls = np.array([
        ['ra_error',         'ra_dec_corr',             'ra_parallax_corr',
                'ra_pmra_corr',       'ra_pmdec_corr',       None],
        ['ra_dec_corr',      'dec_error',               'dec_parallax_corr',
                'dec_pmra_corr',      'dec_pmdec_corr',      None],
        ['ra_parallax_corr', 'dec_parallax_corr',       'parallax_error',
                'parallax_pmra_corr', 'parallax_pmdec_corr', None],
        ['ra_pmra_corr',     'dec_pmra_corr',           'parallax_pmra_corr',
                 'pmra_error',        'pmra_pmdec_corr',     None],
        ['ra_pmdec_corr',    'dec_pmdec_corr',          'parallax_pmdec_corr',
                 'pmra_pmdec_corr',   'pmdec_error',         None],
        [None,               None,                      None,
                 None,                 None, 'radial_velocity_error']
    ])

    # Construct an [nstars,6,6] array of identity matrices
    covs = np.zeros((nstars,6,6))
    idx = np.arange(6)
    covs[:, idx, idx] = 1.0

    # Insert correlations into off diagonals
    for i in range(0,5):
        for j in range(i+1,5):
            covs[:,i,j] = covs[:,j,i] = data[cls[i,j]]

    # multiply each row and each column by appropriate error
    for i in range(6):
        covs[:,i,:] *= np.tile(data[cls[i,i]], (6,1)).T
        covs[:,:,i] *= np.tile(data[cls[i,i]], (6,1)).T

    # Might need to introduce some artificial uncertainty in
    # ra and dec so as to avoid indefinite matrices (too narrow)

    # RA and DEC errors are actually quoted in mas, so we convert cov entries
    # into degrees
    covs[:,:,:2] /= 3600000.
    covs[:,:2,:] /= 3600000.

    return means, covs

def convertGaiaToXYZUVWDict(astr_file="gaia_dr2_ok_plx", server=False,
                            return_dict=False):
    """
    Supposed to generate XYZYVW dictionary for input to GroupFitter

    Doesn't work on whole Gaia catalogue... too much memory I think
    """
    if server:
        rdir = '/data/mash/tcrun/'
    else:
        rdir = '../data/'

    gaia_astr_file = rdir+astr_file+".fits"
    logging.info("Converting: {}".format(gaia_astr_file))
    hdul = fits.open(gaia_astr_file)#, memmap=True)
    logging.info("Loaded hdul")
    means, covs = convertManyRecToArray(hdul[1].data)
    logging.info("Converted many recs")
    astr_dict = {'astr_mns': means, 'astr_covs': covs}
    cv.convertMeasurementsToCartesian(
        astr_dict=astr_dict, savefile=rdir+astr_file+"_xyzuvw.fits")
    logging.info("Converted and saved dictionary")
    if return_dict:
        return {'xyzuvw':means, 'xyzuvw_cov':covs}


def convertGaiaMeansToXYZUVW(astr_file="all_rvs_w_ok_plx", server=False):
    """
    Generate mean XYZUVW for eac star in provided fits file (with Gaia format)
    """
    if server:
        rdir = '/data/mash/tcrun/'
    else:
        rdir = '../data/'
    #gaia_astr_file = rdir+'all_rvs_w_ok_plx.fits'
    gaia_astr_file = rdir+astr_file+".fits"
    hdul = fits.open(gaia_astr_file)#, memmap=True)
    nstars = hdul[1].data.shape[0]
    means = np.zeros((nstars,6))
    means[:,0] = hdul[1].data['ra']
    means[:,1] = hdul[1].data['dec']
    means[:,2] = hdul[1].data['parallax']
    means[:,3] = hdul[1].data['pmra']
    means[:,4] = hdul[1].data['pmdec']
    means[:,5] = hdul[1].data['radial_velocity']

    xyzuvw_mns = cc.convertManyAstrometryToLSRXYZUVW(means, mas=True)
    np.save(rdir + astr_file + "mean_xyzuvw.npy", xyzuvw_mns)