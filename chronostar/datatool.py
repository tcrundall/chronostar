from __future__ import print_function, division
"""
A suite of useful functions for partitioning the gaia data
TODO: write tests especially for cov construction
"""

from astropy.io import fits
from astropy.table import Table
import numpy as np
import logging
from scipy import stats
import sys
sys.path.insert(0, '..')

import chronostar.converter as cv
import chronostar.coordinate as cc


def gauss(x, mu, sig):
    """
    Evaluates a 1D gaussian at `x` given mean `mu` and std `sig`
    """
    return 1./(sig*np.sqrt(2*np.pi)) * np.exp(-(x - mu)**2 / (2.*sig**2))

def mvGauss(x, mean, cov_matrix, amp=1.):
    """
    Evaluates a 6D gaussian at `x` given a mean, covariance matrix and amplitude
    """
    dim = 6.
    coeff = np.linalg.det(cov_matrix * 2*np.pi)**(-0.5)
    inv_cm = np.linalg.inv(cov_matrix)
    expon = -0.5 * np.dot(
        x-mean,
        np.dot(inv_cm, x-mean)
    )
    return amp * coeff * np.exp(expon)

def loadGroups(groups_file):
    """A simple utility function to standardise loading of group savefiles

    :param groups_file:
    :return:
    """
    groups = np.load(groups_file)
    if len(groups.shape) == 0:
        groups = np.array([groups.item()])
    return groups


def loadXYZUVW(xyzuvw_file, assoc_name=None):
    """Load mean and covariances of stars in XYZUVW space from fits file

    Parameters
    ----------
    xyzuvw_file : (String)
        Ideally *.fits, the file name of the fits file with and hdulist:
            [1] : xyzuvw
            [2] : xyzuvw covariances

    assoc_name : (String)
        If loading from the Banyan table, can include an association name
        (see docstring for loadDictFromTable for exhaustive list)

    Returns
    -------
    xyzuvw_dict : (dictionary)
        xyzuvw : ([nstars, 6] float array)
            the means of the stars
        xyzuvw_cov : ([nstars, 6, 6] float array)
            the covariance matrices of the stars
    """
    if (xyzuvw_file[-3:] != 'fit') and (xyzuvw_file[-4:] != 'fits'):
        xyzuvw_file = xyzuvw_file + ".fits"

    try:
        # TODO Ask Mike re storing fits files as float64 (instead of '>f8')
        xyzuvw_now = fits.getdata(xyzuvw_file, 1).\
            astype('float64') #hdulist[1].data
        xyzuvw_cov_now = fits.getdata(xyzuvw_file, 2)\
            .astype('float64') #hdulist[2].data
        xyzuvw_dict = {'xyzuvw':xyzuvw_now, 'xyzuvw_cov':xyzuvw_cov_now}
    except:
        # data stored in simply astropy table
        return loadDictFromTable(xyzuvw_file, assoc_name=assoc_name)
    try:
        times = fits.getdata(xyzuvw_file, 3)
        xyzuvw_dict['times'] = times
    except:
        logging.info("No times in fits file")
    try:
        stars_table = fits.getdata(xyzuvw_file, 3)
        xyzuvw_dict['table'] = stars_table
    except:
        logging.info("No table in fits file")
    logging.info("Floats stored in format {}".\
                 format(xyzuvw_dict['xyzuvw'].dtype))
    return xyzuvw_dict

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
    if filename[-4:] != "fits":
        filename += ".fits"
    gaia_file = "../data/all_rvs_w_ok_plx.fits"
    with fits.open(gaia_file) as hdul:
        primary_hdu = fits.PrimaryHDU(header=hdul[1].header)
        hdu = fits.BinTableHDU(data=hdul[1].data[mask])
        new_hdul = fits.HDUList([primary_hdu, hdu])
        new_hdul.writeto(filename, overwrite=True)


# def tempCreateSubFitsFile(data, filename):
#     """
#     Provide a mask (constructed based on Gaia DR2 fits) to build new one
#
#     Parameters
#     ----------
#     mask : [nstars] int array in tuple
#         The output of np.where applying some filter to gaia data
#         e.g.
#             np.where(hdul[1].data[:,1] > 0)
#         produces a mask to grab all stars with positive DEC
#     filename : string
#         name of destination fits file
#     """
#     if filename[-4:] != "fits":
#         filename += ".fits"
#     gaia_file = "../data/all_rvs_w_ok_plx.fits"
#     with fits.open(gaia_file) as hdul:
#         primary_hdu = fits.PrimaryHDU(header=hdul[1].header)
#         hdu = fits.BinTableHDU(data=hdul[1].data[mask])
#         new_hdul = fits.HDUList([primary_hdu, hdu])
#         new_hdul.writeto(filename, overwrite=True)

def calcMedAndSpan(chain, perc=34, sphere=True):
    """
    Given a set of aligned samples, calculate the 50th, (50-perc)th and
     (50+perc)th percentiles.

    Parameters
    ----------
    chain : [nwalkers, nsteps, npars]
        The chain of samples (in internal encoding)
    perc: integer {34}
        The percentage from the midpoint you wish to set as the error.
        The default is to take the 16th and 84th percentile.
    sphere: Boolean {True}
        Currently hardcoded to take the exponent of the logged
        standard deviations. If sphere is true the log stds are at
        indices 6, 7. If sphere is false, the log stds are at
        indices 6:10.

    Returns
    -------
    _ : [npars,3] float array
        For each paramter, there is the 50th, (50+perc)th and (50-perc)th
        percentiles
    """
    npars = chain.shape[-1]  # will now also work on flatchain as input
    flat_chain = np.reshape(chain, (-1, npars))

    # conv_chain = np.copy(flat_chain)
    # if sphere:
    #     conv_chain[:, 6:8] = np.exp(conv_chain[:, 6:8])
    # else:
    #     conv_chain[:, 6:10] = np.exp(conv_chain[:, 6:10])

    # return np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
    #                     zip(*np.percentile(conv_chain,
    #                                        [50-perc,50,50+perc],
    #                                        axis=0))))
    return np.array(map(lambda v: (v[1], v[2], v[0]),
                        zip(*np.percentile(flat_chain,
                                           [50-perc, 50, 50+perc],
                                           axis=0))))


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


def convertRecToArray(sr):
    """
    Inflexbile approach to read in a single row of astrometric data
    and build mean and covraiance matrix

    Looks first for 'radial_velocity_best', and then for
    'radial_velocity'

    Parameters
    ----------
    sr : a row from table
    """
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
    try:
        rv = sr['radial_velocity_best']
        e_rv = sr['radial_velocity_error_best']
    except KeyError:
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


def appendCartColsToTable(table):
    """
    Insert empty place holder columns for cartesian values
    """
    nrows = len(table)
    empty_col = np.array(nrows * [np.nan])
    cart_col_names = ['X', 'Y', 'Z', 'U', 'V', 'W',
                      'dX', 'dY', 'dZ', 'dU', 'dV', 'dW',
                      'c_XY', 'c_XZ', 'c_XU', 'c_XV', 'c_XW',
                              'c_YZ', 'c_YU', 'c_YV', 'c_YW',
                                      'c_ZU', 'c_ZV', 'c_ZW',
                                              'c_UV', 'c_UW',
                                                      'c_VW']
    units = 3*['pc'] + 3*['km/s'] + 3*['pc'] + 3*['km/s'] + 15*[None]
    for ix, col_name in enumerate(cart_col_names):
        table[col_name] = empty_col
        table[col_name].unit = units[ix]
    return


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

def convertGaiaToXYZUVWDict(astr_file="../data/gaia_dr2_ok_plx.fits",
                            server=False,
                            return_dict=False):
    """
    Supposed to generate XYZYVW dictionary for input to GroupFitter

    Doesn't work on whole Gaia catalogue... too much memory I think

    TODO: Sort out a more consistent way to handle file names...
    """
    if server:
        rdir = '/data/mash/tcrun/'
    else:
        rdir = '../data/'

    logging.info("Converting: {}".format(astr_file))
    hdul = fits.open(astr_file)#, memmap=True)
    logging.info("Loaded hdul")
    means, covs = convertManyRecToArray(hdul[1].data)
    logging.info("Converted many recs")
    astr_dict = {'astr_mns': means, 'astr_covs': covs}
    cart_dict = cv.convertMeasurementsToCartesian(
        astr_dict=astr_dict, savefile=rdir+astr_file[:-5]+"_xyzuvw.fits")
    logging.info("Converted and saved dictionary")
    if return_dict:
        return cart_dict


def convertGaiaMeansToXYZUVW(astr_file="all_rvs_w_ok_plx", server=False):
    """
    Generate mean XYZUVW for eac star in provided fits file (Gaia format)
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


def buildMeanAndCovMatFromRow(row):
    """
    Build a covariance matrix from a row

    Paramters
    ---------
    row : astropy Table row
        Entries: {X, Y, Z, U, V, W, dX, dY, ..., cXY, cXZ, ...}

    Return
    ------
    cov_mat : [6,6] numpy array
        Diagonal elements are dX^2, dY^2, ...
        Off-diagonal elements are cXY*dX*dY, cXZ*dX*dZ, ...
    """
    dim = 6
    CART_COL_NAMES = ['X', 'Y', 'Z', 'U', 'V', 'W',
                      'dX', 'dY',   'dZ',   'dU',   'dV',   'dW',
                            'c_XY', 'c_XZ', 'c_XU', 'c_XV', 'c_XW',
                                    'c_YZ', 'c_YU', 'c_YV', 'c_YW',
                                            'c_ZU', 'c_ZV', 'c_ZW',
                                                    'c_UV', 'c_UW',
                                                            'c_VW']
    mean = np.zeros(dim)
    for i, col_name in enumerate(CART_COL_NAMES[:6]):
        mean[i] = row[col_name]

    std_vec = np.zeros(dim)
    for i, col_name in enumerate(CART_COL_NAMES[6:12]):
        std_vec[i] = row[col_name]
    corr_tri = np.zeros((dim,dim))
    # Insert upper triangle (top right) correlations
    for i, col_name in enumerate(CART_COL_NAMES[12:]):
        corr_tri[np.triu_indices(dim,1)[0][i],np.triu_indices(dim,1)[1][i]]\
            =row[col_name]
    # Build correlation matrix
    corr_mat = np.eye(6) + corr_tri + corr_tri.T
    # Multiply through by standard deviations
    cov_mat = corr_mat * std_vec * std_vec.reshape(6,1)
    return mean, cov_mat


def isInAssociation(element, iterable):
    """
    A wrapper for 'is in' which returns false if `iterable` is None
    """
    if iterable is None:
        return False
    if type(iterable) is str:
        return element == iterable
    return element in iterable


def loadDictFromTable(table, assoc_name=None):
    """
    Takes the data in the table, builds dict with array of mean and cov matrices

    Paramters
    ---------
    table : Astropy.Table (or str)

    assoc_name : str
        One of the labels in Moving group column:
         118 Tau, 32 Orionis, AB Doradus, Carina, Carina-Near, Columba,
         Coma Ber, Corona Australis, Hyades, IC 2391, IC 2602,
         Lower Centaurus-Crux, Octans, Platais 8, Pleiades, TW Hya, Taurus,
         Tucana-Horologium, Upper Centaurus Lupus, Upper CrA, Upper Scorpius,
         Ursa Major, beta Pictoris, chi{ 1 For (Alessi 13), epsilon Cha,
         eta Cha, rho Ophiuci

    Returns
    -------
    dict :
        xyzuvw : [nstars,6] array of xyzuvw means
        xyzuvw_cov : [nstars,6,6] array of xyzuvw covariance matrices
        file_name : str
            if table loaded from file, the pathway is stored here
        indices : [nstars] int array
            the table row indices of data converted into arrays
        gaia_ids : [nstars] int array
            the gaia ids of stars successfully converted
    """

    star_pars = {}
    if type(table) is str:
        file_name = table
        star_pars['file_name'] = file_name
        table = Table.read(table)

    xyzuvw = []
    xyzuvw_cov = []
    indices = []
    gaia_ids = []
    nrows = len(table['source_id'])
    for ix, row in enumerate(table):
        if (np.isfinite(row['U']) and
            isInAssociation(row['Moving group'], assoc_name)):
            mean, cov = buildMeanAndCovMatFromRow(row)
            xyzuvw.append(mean)
            xyzuvw_cov.append(cov)
            indices.append(ix)
            gaia_ids.append(row['source_id'])

    star_pars['xyzuvw']     = np.array(xyzuvw).astype(np.float64)
    star_pars['xyzuvw_cov'] = np.array(xyzuvw_cov).astype(np.float64)
    star_pars['indices']    = indices
    star_pars['gaia_ids']   = gaia_ids
    star_pars['table']      = table
    return star_pars


def calcHistogramDensity(x, bin_heights, bin_edges):
    """
    Calculates the density of a pdf characterised by a histogram at point x
    """
    # Check if handling 1D histogram
    if len(bin_heights.shape) == 1:
        raise UserWarning
    dims = len(bin_heights.shape)
    bin_widths = [bins[1] - bins[0] for bins in bin_edges]
    bin_area = np.prod(bin_widths)

    x_ix = tuple([np.digitize(x[dim], bin_edges[dim]) - 1
                  for dim in range(dims)])
    return bin_heights[x_ix] / bin_area


def collapseHistogram(bin_heights, dim):
    """
    Collapse a multi dimensional histogram into a single dimension

    Say you have a 6D histogram, but want the 1D projection onto
    the X axis, simply call this function with `dim=0`
    Uses string trickery and einstein sum notation to pick the
    dimension to retain


    Parameters
    ----------
    bin_heights: `dims`*[nbins] array
        an array of length `dims` with shape (nbins, nbins, ..., nbins)
        first output of np.histogramdd
    dim: integer
        the dimension which to collapse onto
    """
    dims = len(bin_heights.shape)

    # generates `dims` length string 'ijkl...'
    indices = ''.join([chr(105+i) for i in range(dims)])

    # reduce-sum each axis except the `dim`th axis
    summed_heights = np.einsum('{}->{}'.format(indices, indices[dim]),
                               bin_heights)
    return summed_heights


def sampleHistogram(bin_heights, bin_edges, lower_bound, upper_bound,
                    npoints=10):
    """
    Hard coded to return 1D projection to x
    """
    # x_vals = np.linspace(lower_bound[0], upper_bound[0], npoints, endpoint=False)
    # step_sizes = (upper_bound - lower_bound)/(npoints-1)
    # # y = lower_bound[1]
    # heights = []
    densities = []
    pts = []
    # TODO: Consider step size not accurate, need npoitns + 1 or something
    # TODO: still not right.. make sure not counting "empty" bins...
    for x in np.linspace(lower_bound[0], upper_bound[0], npoints, endpoint=False):
        print('x: ', x)
        for y in np.linspace(lower_bound[1], upper_bound[1], npoints, endpoint=False):
            print('y: ', y)
            for z in np.linspace(lower_bound[2], upper_bound[2], npoints, endpoint=False):
                for u in np.linspace(lower_bound[3], upper_bound[3], npoints, endpoint=False):
                    for v in np.linspace(lower_bound[4], upper_bound[4], npoints, endpoint=False):
                        for w in np.linspace(lower_bound[5], upper_bound[5], npoints, endpoint=False):
                            pt = [x,y,z,u,v,w]
                            pts.append(pt)
                            density = calcHistogramDensity(pt,
                                                           bin_heights,
                                                           bin_edges)
                            densities.append(density)
                            # only product by area of remaining dimensions
                            # height += density * np.prod(step_sizes[1:])
        # heights.append(height)

    return np.array(pts), np.array(densities)

def integrateHistogram2(bin_heights, bin_edges, lower_bound, upper_bound,
                       dim):
    """
    Hard coded to return 1D projection to x
    """
    npoints = 10
    dims = len(lower_bound)
    pts, densities = sampleHistogram(bin_heights, bin_edges, lower_bound,
                                     upper_bound, npoints=npoints)
    new_bin_edges = np.linspace(lower_bound[dim], upper_bound[dim], npoints,
                                endpoint=False)
    step_sizes = (upper_bound - lower_bound) / (npoints)

    new_bin_heights = []
    for ledge in new_bin_edges:
        height = np.sum(densities[np.where(pts[:,dim] == ledge)])
        new_bin_heights.append(height * np.prod(step_sizes) / step_sizes[dim])

    return np.array(new_bin_heights), np.array(new_bin_edges)

    #
    # xs, ys, zs, us, vs, ws =\
    #     np.meshgrid(np.arange(lower_bound[0], upper_bound[0], npoints),
    #                 np.arange(lower_bound[1], upper_bound[1], npoints),
    #                 np.arange(lower_bound[2], upper_bound[2], npoints),
    #                 np.arange(lower_bound[3], upper_bound[3], npoints),
    #                 np.arange(lower_bound[4], upper_bound[4], npoints),
    #                 np.arange(lower_bound[5], upper_bound[5], npoints),
    #                 )
    # np.mgrid[lower_bound[0]:upper_bound[0]:10j,
    #         lower_bound[1]:upper_bound[1]:10j,
    #         lower_bound[2]:upper_bound[2]:10j,
    #         lower_bound[3]:upper_bound[3]:10j,
    #         lower_bound[4]:upper_bound[4]:10j,
    #         lower_bound[5]:upper_bound[5]:10j]
    #
    #
    # x_vals = np.linspace(lower_bound[0], upper_bound[0], npoints)
    # step_sizes = (upper_bound - lower_bound)/(npoints-1)
    # y = lower_bound[1]
    # heights = []
    # # TODO: Consider step size not accurate, need npoitns + 1 or something
    # # TODO: still not right.. make sure not counting "empty" bins...
    # for x in x_vals:
    #     height = 0
    #     for y in np.linspace(lower_bound[1], upper_bound[1], npoints):
    #         for z in np.linspace(lower_bound[2], upper_bound[2], npoints):
    #             for u in np.linspace(lower_bound[3], upper_bound[3], npoints):
    #                 for v in np.linspace(lower_bound[4], upper_bound[4], npoints):
    #                     for w in np.linspace(lower_bound[5], upper_bound[5], npoints):
    #                         density = calcHistogramDensity([x,y,z,u,v,w],
    #                                                        bin_heights,
    #                                                        bin_edges)
    #                         # only product by area of remaining dimensions
    #                         height += density * np.prod(step_sizes[1:])
    #     heights.append(height)
    # return x_vals, heights


def samplePoints(bin_heights, bin_edges, lower_bound, upper_bound):
    """
    Montecarlo sample points within range of bounds,
    :param bin_heights:
    :param bin_edges:
    :param lower_bound:
    :param upper_bound:
    :return:
    """


def sampleAndBuild1DHist(bin_heights, bin_edges, lower_bound, upper_bound,
                         dim):
    """
    Monte carlo sample master histogram within bounds, then project to 1D

    :param bin_heights:
    :param bin_edges:
    :param lower_bound:
    :param upper_bound:
    :param dim:
    :return:
    new_bin_heights
    new_bin_edges
    """
    ndims = len(bin_heights.shape)
    nsamples = 10**6
    samples = samplePoints(bin_heights, bin_edges, lower_bound, upper_bound)
    sample_hist = np.histogramdd(samples)

    #TODO: normalise by nsamples_survived, and renormalise by star count
    return collapseHistogram(sample_hist[0], dim), sample_hist[1][dim]


def getDensity(point, data, bin_per_std=8.):
    """
    Given a point in 6D space, and some data distributed in said space,
    get the density at that point.
    """
    point = np.array(point)
    # anchor the offset from the standard deviation of the data
    spread = np.std(data, axis=0) / float(bin_per_std)
    # print("spread: ", spread)
    ubound = point + 0.5*spread
    lbound = point - 0.5*spread

    nstars = len(
        np.where(np.all((data < ubound) & (data > lbound), axis=1))[0]
    )

    volume = np.prod(spread)
    return nstars / volume


def getZfromOrigins(origins, star_pars):
    if type(origins) is str:
        origins = loadGroups(origins)
    if type(star_pars) is str:
        star_pars = loadXYZUVW(star_pars)
    nstars = star_pars['xyzuvw'].shape[0]
    ngroups = len(origins)
    nassoc_stars = np.sum([o.nstars for o in origins])
    using_bg = nstars != nassoc_stars
    z = np.zeros((nstars, ngroups + using_bg))
    stars_so_far = 0
    # set associaiton members memberships to 1
    for i, o in enumerate(origins):
        z[stars_so_far:stars_so_far+o.nstars, i] = 1.
        stars_so_far += o.nstars
    # set remaining stars as members of background
    if using_bg:
        z[stars_so_far:,-1] = 1.
    return z


def getKernelDensities(data, points, get_twins=False):
    """
    Build a PDF from `data`, then evaluate said pdf at `points`

    Parameters
    ----------
    data : [nstars, 6] array of star means (typically data/gaia_xyzuvw.npy content)
    points: [npoints, 6] array of star means of particular interest
    """
    if type(data) is str:
        data = np.load(data)
    nstars = data.shape[0]

    kernel = stats.gaussian_kde(data.T)

    bg_ln_ols = np.log(nstars)+kernel.logpdf(points.T)

    if get_twins:
        twin_points = np.copy(points)
        twin_points[:,2] *= -1
        twin_points[:,5] *= -1

        twin_bg_ln_ols = np.log(nstars)+kernel.logpdf(twin_points.T)
        return bg_ln_ols, twin_bg_ln_ols
    else:
        return bg_ln_ols
