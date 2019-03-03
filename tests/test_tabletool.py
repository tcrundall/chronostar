"""
Test a bunch of functions that serve as an interface to standard stellar data
table
"""
import numpy as np

from astropy.io import fits
import sys
sys.path.insert(0, '..')

from chronostar.synthdata import SynthData
from chronostar import tabletool
from chronostar import datatool
from chronostar import converter as cv


def convertManyRecToArray(data):
    """
    Convert many Fits Records in astrometry into mean and covs (astro)

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
    nstars = data.shape[0]
    means = np.zeros((nstars,6))

    means[:,0] = data['ra']
    means[:,1] = data['dec']
    means[:,2] = data['parallax']
    means[:,3] = data['pmra']
    means[:,4] = data['pmdec']
    means[:,5] = data['radial_velocity']

    # Array of dictionary keys to aid construction of cov matrices
    cls = np.array([
        ['ra_error',         'ra_dec_corr',           'ra_parallax_corr',
                'ra_pmra_corr',       'ra_pmdec_corr',       None],
        ['ra_dec_corr',      'dec_error',             'dec_parallax_corr',
                'dec_pmra_corr',      'dec_pmdec_corr',      None],
        ['ra_parallax_corr', 'dec_parallax_corr',     'parallax_error',
                'parallax_pmra_corr', 'parallax_pmdec_corr', None],
        ['ra_pmra_corr',     'dec_pmra_corr',         'parallax_pmra_corr',
                 'pmra_error',        'pmra_pmdec_corr',     None],
        ['ra_pmdec_corr',    'dec_pmdec_corr',        'parallax_pmdec_corr',
                 'pmra_pmdec_corr',   'pmdec_error',         None],
        [None,               None,                     None,
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

    # RA and DEC errors are actually quoted in mas, so we convert cov
    # entries into degrees
    covs[:,:,:2] /= 3600000.
    covs[:,:2,:] /= 3600000.

    return means, covs

# Retired function put here for comparision reasons
def convertGaiaToXYZUVWDict(astr_file="../data/gaia_dr2_ok_plx.fits",
                            server=False,
                            return_dict=False):
    """
    Supposed to generate XYZYVW dictionary for input to GroupFitter

    Doesn't work on whole Gaia catalogue... too much memory I think

    TODO: Sort out a more consistent way to handle file names...
    """
    hdul = fits.open(astr_file)#, memmap=True)
    means, covs = convertManyRecToArray(hdul[1].data)
    astr_dict = {'astr_mns': means, 'astr_covs': covs}
    cart_dict = cv.convertMeasurementsToCartesian(astr_dict=astr_dict)

    return cart_dict


def alternativeBuildCovMatrix(data):
    nstars = len(data)
    cls = np.array([
        ['X', 'c_XY', 'c_XZ', 'c_XU', 'c_XV', 'c_XW'],
        ['c_XY', 'Y', 'c_YZ', 'c_YU', 'c_YV', 'c_YW'],
        ['c_XZ', 'c_YZ', 'Z', 'c_ZU', 'c_ZV', 'c_ZW'],
        ['c_XU', 'c_YU', 'c_ZU', 'U', 'c_UV', 'c_UW'],
        ['c_XV', 'c_YV', 'c_ZV', 'c_UV', 'V', 'c_VW'],
        ['c_XW', 'c_YW', 'c_ZW', 'c_UW', 'c_VW', 'W'],
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

    return covs


def test_convertTableXYZUVWToArray():
    """
    TODO: replace file with synthetic data once XYZUVW conversion is implemented
    """
    PARS = np.array([
        [0., 0., 0., 0., 0., 0., 10., 5., 1e-5],
        [5., 0., -5., 0., 0., 0., 10., 5., 40.]
    ])
    STARCOUNTS = [50, 30]
    COMP_FORMS = 'sphere'

    filename = '../data/beta_Pictoris_with_gaia_small_everything_final.fits'

    orig_star_pars = datatool.loadDictFromTable(filename)
    means, covs = tabletool.convertTableXYZUVWToArray(
        orig_star_pars['table'][orig_star_pars['indices']]
    )

    assert np.allclose(orig_star_pars['xyzuvw'], means)
    assert np.allclose(orig_star_pars['xyzuvw_cov'], covs)

