"""
Test a bunch of functions that serve as an interface to standard stellar data
table
"""
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.units.core import UnitConversionError
try:
    import exceptions
except ImportError:
    import builtins as exceptions # python 3 consistent
import sys
sys.path.insert(0, '..')

from chronostar.synthdata import SynthData
from chronostar import tabletool
from chronostar import datatool
from chronostar import converter as cv


# Retired funciton, put here for comparison reasons
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
    cov_labels = np.array([
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
            covs[:,i,j] = covs[:,j,i] = data[cov_labels[i,j]]

    # multiply each row and each column by appropriate error
    for i in range(6):
        covs[:,i,:] *= np.tile(data[cov_labels[i,i]], (6,1)).T
        covs[:,:,i] *= np.tile(data[cov_labels[i,i]], (6,1)).T

    return covs


def test_convertTableXYZUVWToArray():
    """
    TODO: replace file with synthetic data once XYZUVW conversion is implemented
    """
    filename = '../data/beta_Pictoris_with_gaia_small_everything_final.fits'

    orig_star_pars = datatool.loadDictFromTable(filename)
    main_colnames, error_colnames, corr_colnames =\
        tabletool.getHistoricalCartColnames()
    means, covs = tabletool.buildDataFromTable(
            orig_star_pars['table'][orig_star_pars['indices']],
            main_colnames=main_colnames,
            error_colnames=error_colnames,
            corr_colnames=corr_colnames
    )

    assert np.allclose(orig_star_pars['xyzuvw'], means)
    assert np.allclose(orig_star_pars['xyzuvw_cov'], covs)

def test_convertSynthTableToCart():
    PARS = np.array([
        [0., 0., 0., 0., 0., 0., 10., 5., 1e-10],
        # [5., 0., -5., 0., 0., 0., 10., 5., 40.]
    ])
    STARCOUNTS = [50] #, 30]
    COMP_FORMS = 'sphere'

    filename = 'temp_data/full_synth_data.fits'
    synth_data = SynthData(pars=PARS, starcounts=STARCOUNTS,
                           comp_forms=COMP_FORMS, measurement_error=1e-3
                           )
    synth_data.synthesiseEverything()
    synth_data.storeTable(filename=filename, overwrite=True)

    table = tabletool.convertTableAstroToXYZUVW(filename, return_table=True)

    assert np.allclose(table['x0'], table['X'])


def test_convertAstrTableToCart():
    """
    Using a historical table, confirm that cartesian conversion yields
    same results
    """
    filename = '../data/beta_Pictoris_with_gaia_small_everything_final.fits'
    table = Table.read(filename)
    # Drop stars that have gone through any binary checking
    table = Table(table[100:])
    # Manually change different colnames
    table['radial_velocity'] = table['radial_velocity_best']
    table['radial_velocity_error'] = table['radial_velocity_error_best']

    # load in original means and covs
    # orig_astr_means, orig_astr_covs =\
    #     tabletool.buildDataFromTable(table=table, cartesian=False)
    orig_cart_means, orig_cart_covs =\
        tabletool.buildDataFromTable(table=table, cartesian=True,
                                     historical=True)

    tabletool.convertTableAstroToXYZUVW(table=table, write_table=False)

    cart_means, cart_covs = tabletool.buildDataFromTable(table, cartesian=True)

    assert np.allclose(orig_cart_means, cart_means)
    assert np.allclose(table['dX'], table['X_error'])
    assert np.allclose(orig_cart_covs, cart_covs)


def test_badColNames():
    main_colnames, error_colnames, corr_colnames = \
        tabletool.getColnames(cartesian=False)

    main_colnames[5] = 'radial_velocity_best'
    error_colnames[5] = 'radial_velocity_error_best'
    # corrupt ordering of column names
    corrupted_error_colnames = list(error_colnames)
    corrupted_error_colnames[0], corrupted_error_colnames[3] =\
        error_colnames[3], error_colnames[0]

    filename = '../data/beta_Pictoris_with_gaia_small_everything_final.fits'
    table = Table.read(filename)

    # Only need a handful of rows
    table = Table(table[:10])

    # Catch when units are inconsistent
    try:
        tabletool.convertTableAstroToXYZUVW(table,
                                            main_colnames=main_colnames,
                                            error_colnames=corrupted_error_colnames,
                                            corr_colnames=corr_colnames)
    except Exception as e:
        assert type(e) == exceptions.UserWarning

    # In the case where units have not been provided, then just leave it be
    try:
        error_colnames[0] = 'ra_dec_corr'
        tabletool.convertTableAstroToXYZUVW(table,
                                            main_colnames=main_colnames,
                                            error_colnames=error_colnames,
                                            corr_colnames=corr_colnames)
    except:
        assert False

if __name__ == '__main__':
    test_badColNames()
