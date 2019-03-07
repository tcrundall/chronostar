"""
A suite of useful functions for partitioning the gaia data

Look here for all your table reading needs.

Also includes a class for abstracting the parametrisation of a Gaussian
origin.

TODO: write tests especially for cov construction
"""

from __future__ import print_function, division

from astropy.io import fits
from astropy.table import Table
import numpy as np
import logging
from scipy import stats

# from . import converter as cv
from chronostar import coordinate as cc


def gauss(x, mu, sig):
    """
    Evaluates a 1D gaussian at `x` given mean `mu` and std `sig`

    Parameters
    ----------
    x : float
        point at which to evaluate Gaussian
    mu : float
        mean of Gaussian
    sig : float
        standard deviation of Gaussian

    Returns
    -------
    result : float
        evaluation of 1D Gaussian at `x`
    """
    return 1./(sig*np.sqrt(2*np.pi)) * np.exp(-(x - mu)**2 / (2.*sig**2))


def mvGauss(x, mean, cov_matrix, amp=1.):
    """
    Evaluates a 'dim' dimension gaussian at `x` given a mean,
    covariance matrix and amplitude

    Parameters
    ----------
    x : [dim] float array
        point at which to evaluate multivariate Gaussian
    mean : [dim] float array
        mean of the multivariate Gaussian
    cov_matrix : [dim, dim] array
        covariance matrix of the multivariate Gaussian
    amp : float {1.}
        amplitude of the multivariate Gaussian

    Returns
    -------
    result : float
        evaluation of the multivariate gaussian at point `x`
    """
    coeff = np.linalg.det(cov_matrix * 2*np.pi)**(-0.5)
    inv_cm = np.linalg.inv(cov_matrix)
    expon = -0.5 * np.dot(
        x-mean,
        np.dot(inv_cm, x-mean)
    )
    return amp * coeff * np.exp(expon)


def loadGroups(groups_file):
    """A simple utility function to standardise loading of group savefiles

    Ensures consistency by returning groups as an array of synthesiser.Group
    objects, even if only one Group object is loaded.

    Parameters
    ----------
    groups_file : string
        Path to savefile of file. File must be result of np.save(groups)
        where 'groups' is either a list (or a single instance) of
        synthesiser.Group objects

    Returns
    -------
    groups : [synthesiser.Group]
        A list of Group objects
    """
    groups = np.load(groups_file)
    if len(groups.shape) == 0:
        groups = np.array([groups.item()])
    return groups


def loadXYZUVW(xyzuvw_file, assoc_name=None):
    """Load mean and covariances of stars in XYZUVW space from fits file

    TODO: Streamline for astropy table usage
    TODO: port fits HDUL to separate, dedicated function

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


def getBestSample(chain, lnprob):
    """
    Finds the sample with the highest lnprob

    Parameters
    ----------
    chain : [nsteps, nwalkers, npars] float array
        A chain of walkers from an emcee sampling run
    lnprob : [nsteps, nwalkers] float array
        An array of log probabilities from an emcee sampling run

    Returns
    -------
    result : [9] -or- [15] array
        The (internal) parameters describing the best fitting sample
    """
    npars = chain.shape[-1]
    best_ix = np.argmax(lnprob)
    flat_chain = chain.reshape(-1, npars)
    return flat_chain[best_ix]


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

def calcMedAndSpan(chain, perc=34, intern_to_extern=False, sphere=True):
    """
    Given a set of aligned samples, calculate the 50th, (50-perc)th and
     (50+perc)th percentiles.

    Parameters
    ----------
    chain : [nwalkers, nsteps, npars] array -or- [nwalkers*nsteps, npars] array
        The chain of samples (in internal encoding)
    perc: integer {34}
        The percentage from the midpoint you wish to set as the error.
        The default is to take the 16th and 84th percentile.
    intern_to_extern : boolean {False}
        Set to true if chain has dx and dv provided in log form and output
        is desired to have dx and dv in linear form
    sphere: Boolean {True}
        Currently hardcoded to take the exponent of the logged
        standard deviations. If sphere is true the log stds are at
        indices 6, 7. If sphere is false, the log stds are at
        indices 6:10.

    Returns
    -------
    result : [npars,3] float array
        For each parameter, there is the 50th, (50+perc)th and (50-perc)th
        percentiles
    """
    npars = chain.shape[-1]  # will now also work on flatchain as input
    flat_chain = np.reshape(chain, (-1, npars))

    if intern_to_extern:
        # take the exponent of the log(dx) and log(dv) values
        flat_chain = np.copy(flat_chain)
        if sphere:
            flat_chain[:, 6:8] = np.exp(flat_chain[:, 6:8])
        else:
            flat_chain[:, 6:10] = np.exp(flat_chain[:, 6:10])

    return np.array(map(lambda v: (v[1], v[2], v[0]),
                        zip(*np.percentile(flat_chain,
                                           [50-perc, 50, 50+perc],
                                           axis=0))))


def convertRecToArray(sr):
    """
    Take a single row from astrometry table and build covariance matrix

    Builds a central estimate (mean) and covariance matrix with field order:
    ra, dec, parallax, proper-motion ra, proper-motion dec, radial velocity.
    Currently hardcoded to handle gaia column default names, and cannot handle
    absent correlation values. Looks first for 'radial_velocity_best', and then
    for 'radial_velocity'.

    TODO: Establish whether robustness checks occur here or earlier

    Parameters
    ----------
    sr : table row
        A single row from astrometric table with columns:
        'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error',
        'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
        'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr',
        'ra_pmdec_corr', 'dec_parallax_corr', 'dec_pmra_corr',
        'dec_pmdec_corr', 'parallax_pmra_corr', 'parallax_pmdec_corr',
        'pmra_pmdec_corr',

    Returns
    -------
    mean : [6] float array
        central estimate of full 6D stellar astrometry for single star
    cov : [6, 6] float array
        covariance matrix of full 6D stellar astrometry for single star
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

    try:
        rv = sr['radial_velocity_best']
        e_rv = sr['radial_velocity_error_best']
    except KeyError:
        rv = sr['radial_velocity']
        e_rv = sr['radial_velocity_error']

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

    Parameters
    ----------
    table : astropy.table.Table object
        Modifies table in place by appending empty columns for cartesian
        values. Default values in column are `np.nan`.

    Returns
    -------
    None
    """
    nrows = len(table)
    empty_col = np.array(nrows * [np.nan])
    cart_col_names = ['X', 'dX', 'Y', 'dY', 'Z', 'dZ',
                      'U', 'dU', 'V', 'dV', 'W', 'dW',
                      'c_XY', 'c_XZ', 'c_XU', 'c_XV', 'c_XW',
                              'c_YZ', 'c_YU', 'c_YV', 'c_YW',
                                      'c_ZU', 'c_ZV', 'c_ZW',
                                              'c_UV', 'c_UW',
                                                      'c_VW']
    units = 6*['pc'] + 6*['km/s'] + 15*[None]
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
    A wrapper for 'is in' which returns true if `iterable` is None

    If `iterable` is None, then we accept all elements
    """
    if iterable is None:
        return True
    if type(iterable) is str:
        return element['Moving group'] == iterable
    return element['Moving group'] in iterable


def loadTable(table):
    return Table.read(table)


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

    gaia_col_name = 'source_id'
    if gaia_col_name not in table.colnames:
        gaia_col_name = 'gaia_dr2'


    xyzuvw = []
    xyzuvw_cov = []
    indices = []
    gaia_ids = []
    nrows = len(table[gaia_col_name])
    for ix, row in enumerate(table):
        if nrows > 10000 and ix % 1000==0:
            print("Done {:7} of {} | {:.2}%".format(ix, nrows,
                                                    float(ix)/nrows*100))
        if (np.isfinite(row['U']) and
            isInAssociation(row, assoc_name)):
            mean, cov = buildMeanAndCovMatFromRow(row)
            xyzuvw.append(mean)
            xyzuvw_cov.append(cov)
            indices.append(ix)
            gaia_ids.append(row[gaia_col_name])

    star_pars['xyzuvw']     = np.array(xyzuvw).astype(np.float64)
    star_pars['xyzuvw_cov'] = np.array(xyzuvw_cov).astype(np.float64)
    star_pars['indices']    = np.array(indices)
    star_pars['gaia_ids']   = np.array(gaia_ids)
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


def getZfromOrigins(origins, star_pars=None):
    """
    A simple utility function, reconstruct the true Z from origins for plotting.

    If a background was used, then `star_pars` must be provided, with
    stars listed in the order they were generated. The order of generation
    must follow that of the origins order.

    Parameters
    ----------
    origins: [ngroups] list of Group objects -or-
             [ngroups] list of parameters (any Chronostar form) -or-
             filename to stored Groups
        a list of groups (as synthesiser.Group objects or in parametrisation)
        that were used to generate synthetic stars.
    star_pars: dict -or- filename to stored dict - {None}

    """
    if type(origins) is str:
        origins = loadGroups(origins)
    try:
        nstar_arr = np.array([o.nstars for o in origins])
    except AttributeError:
        nstar_arr = np.array([int(o[-1]) for o in origins])
    nassoc_stars = np.sum(nstar_arr)

    # Determine how many stars came from background.
    # If we don't have star_pars then there's no way of determining how many
    # stars were drawn from a background, so we assume there were none.
    if star_pars is None:
        using_bg = False
        nstars = nassoc_stars
    # If number of provided stars exceeds that of groups, then must have
    # come from background
    else:
        if type(star_pars) is str:
            star_pars = loadXYZUVW(star_pars)
        nstars = star_pars['xyzuvw'].shape[0]
        using_bg = nstars != nassoc_stars

    # Build the membership array, `z`
    # set association members memberships to 1
    ngroups = len(origins)
    z = np.zeros((nstars, ngroups + using_bg))
    stars_so_far = 0
    for i, n in enumerate(nstar_arr):
        z[stars_so_far:stars_so_far+n, i] = 1.
        stars_so_far += n
    # set remaining stars as members of background
    if using_bg:
        z[stars_so_far:,-1] = 1.
    return z


def getKernelDensities(data, points, get_twins=False, amp_scale=1.0):
    """
    Build a PDF from `data`, then evaluate said pdf at `points`

    Changed behaviour (4/12/2018) such that inverts Z and W of points)

    Parameters
    ----------
    data : [nstars, 6] array of star means (typically data/gaia_xyzuvw.npy content)
    points: [npoints, 6] array of star means of particular interest
    """
    if type(data) is str:
        data = np.load(data)
    nstars = amp_scale * data.shape[0]

    kernel = stats.gaussian_kde(data.T)
    points = np.copy(points)
    points[:,2] *= -1
    points[:,5] *= -1

    bg_ln_ols = np.log(nstars)+kernel.logpdf(points.T)

    if get_twins:
        twin_points = np.copy(points)
        twin_points[:,2] *= -1
        twin_points[:,5] *= -1

        twin_bg_ln_ols = np.log(nstars)+kernel.logpdf(twin_points.T)
        return bg_ln_ols, twin_bg_ln_ols
    else:
        return bg_ln_ols
