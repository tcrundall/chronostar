from __future__ import division, print_function

"""
TODO: Come up with a neater way to handle missing data. Maybe fits
will permit blanks to be included, should explore this
"""

import numpy as np
import re
from astropy.table import Table
from astroquery.gaia import Gaia


import sys
sys.path.insert(0, '..')
import chronostar.coordinate as coord
import chronostar.transform as tf

def clean_row(row,i):
    """
    Split row into constituents by manually insert delimiters (based on
    byte separation listed in "machine readable" text file

    Parameters
    ----------
    row : str
        a ~160 character string from the BANYAN XI table 5, machine readable
        format:
        http://iopscience.iop.org.virtual.anu.edu.au/0004-637X/856/1/23/suppdata/apjaaae09t5_mrt.txt
    """
    # convert string
    char_array = np.array(list(row))

    #insert entry dividers, then split by them
    div_ix = (
        np.array([4, 22, 29, 40, 43, 46, 53, 57, 60, 66, 74, 79, 87, 92, 99,
                  105, 113, 120, 126, 133, 139, 146, 152, 158, 163, 168, 180,
                  193, 207])
        # np.array([6, 34, 48, 51, 54, 60, 64, 67, 72, 80, 86, 94, 100,
        #           107, 112, 119, 125, 137, 141, 145, 156]),
    )
    char_array[div_ix] = ','
    new_csv_row = (''.join(char_array)).split(',')

    # remove excess whitespace surrounding data
    new_csv_row = np.array([entry.strip() for entry in new_csv_row])

    return new_csv_row

def readBanyanTxtFile():
    NHEADER_ROWS = 348

    file_name = '../data/banyan_2_data.txt'

    fp = open(file_name, 'r')
    data_raw = fp.readlines()

    data_cleaned = [clean_row(row, i) for (i, row) in enumerate(data_raw) if
                    i > NHEADER_ROWS]
    return data_cleaned


def insertLitRVs(gt, banyan_data):
    # Extract rvs from banyan txt file and insert into extended fits table
    rvs_and_err = np.array([row[13:15] for row in banyan_data])
    blank_mask = np.where(rvs_and_err == '')
    rvs_and_err[blank_mask] = np.nan
    rvs_and_err = rvs_and_err.astype(np.float)
    gt['radial_velocity_lit'] = rvs_and_err[:, 0]
    gt['radial_velocity_error_lit'] = rvs_and_err[:, 1]
    # find entries with no radial velocities from anywhere
    no_best_rv_mask = gt['radial_velocity_best_flag'] == 'NULL'

    # get mask for rvs that can be improved
    # i.e. find existing rvlit entries with rvbest_err < rvlit_err
    imp_mask = np.where(
            ( (gt['radial_velocity_best_flag'] == 'NULL')
                & (np.isfinite(gt['radial_velocity_lit'])))
        | (gt['radial_velocity_error_lit'] < gt['radial_velocity_error_best']))

    gt['radial_velocity_best'][imp_mask] = gt['radial_velocity_lit'][imp_mask]
    gt['radial_velocity_error_best'][imp_mask] =\
        gt['radial_velocity_error_lit'][imp_mask]
    gt['radial_velocity_best_flag'][imp_mask] = 'LIT'


def getMassFromSpectralType(spec):
    """
    Takes as input a string of BANYAN XI table 5 format, and approximates
    mass using Kraus & Hillenbrand (2007)

    K&H list a range of spectral type to mass conversions. If provided
    spec falls in the gap, we linearly interpolate between the two nearest
    values

    Note: if the string states a range, we just take the first type

    Paramters
    ---------
    spec : str
        First char must be a letter from {O, B, A, F, G, A, K}
        Next char(s) must be a numeric. Remaining chars are ignored
        (T type stars are of order 0.01 solar masses and can typically
        be disregarded)

    Returns
    -------
    mass : float
    """
    if spec is None or spec == '':
        return np.nan

    spec_masses = {
        'B' :
            (np.array([8.,10.]),
             np.array([3.8,2.9])),
        'A' :
            (np.array([0.,2.,5.,7.,10.]),
             np.array([2.9,2.4,2.0,1.8,1.6])),
        'F' :
            (np.array([0.,2.,5.,8.,10.]),
             np.array([1.6,1.5,1.25,1.17,1.11])),
        'G' :
            (np.array([0.,2.,5.,8.,10.]),
             np.array([1.11,1.06,1.04,0.98,0.9])),
        'K' :
            (np.array([0.,2.,4.,5.,7.,10.]),
             np.array([0.9,0.82,0.75,0.70,0.63,0.59])),
        'M' :
            (np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]),
             np.array([0.59,0.54,0.42,0.29,0.20,0.15,0.12,
                       0.11,0.102,0.088,0.078])),
        'L' :
            (np.array([0.]),
             np.array([0.078]))
    }

    spec_type = spec[0]
    if spec_type not in spec_masses.keys():
        # If dealing with an 'O' star, likely to be dominant source
        # of mass in system, so return some large mass
        if spec_type == 'O':
            return 100.
        # Otherwise, star is likely to be negligible factor
        else:
            return 0.

    # find first float-ish number in string
    try:
        matches = re.findall("\d+[\.\d+]*", spec)
        if len(matches) == 0:
            spec_val = 0.
        else:
            spec_val = float(matches[0])

    except:
        print(spec)
        import pdb
        pdb.set_trace()


    try:
        mass = np.interp(
            spec_val,
            spec_masses[spec_type][0],
            spec_masses[spec_type][1],
        )
    except IndexError:
        import pdb; pdb.set_trace()
    return mass

if __name__=='__main__':
    """Read in banyan text file, and Marusa's fits table, insert missing
    literature rvs and update 'best rv' column"""

    print("Beginning script")

    # Load data tables
    banyan_data = readBanyanTxtFile()
    star_names = np.array([row[1] for row in banyan_data])
    mg_names = np.array([row[2] for row in banyan_data])
    bpmg_data = np.array(banyan_data)[np.where(mg_names == 'BPMG')]

    # gaia dr2 ids, manually taken from Gaia archive by individually
    # resolving star names
    gaia_ids = [
        3292922293081928192,
        3181961503752885248,
        3290081910949989888, # slightly brighter
        3290081906654767616, # companion of ^^
        3209947441933983744,
        2477815222028038272,
        4093006560668685568, # proper motion needed to confirm
        3393207610483520896,
        5924485966955008896,
        6631762764424312960,
    ]

    astro_query = False
    if astro_query:
        query = '''
        SELECT gaia.*
        FROM gaiadr2.gaia_source AS gaia
        WHERE gaia.source_id IN %s
        ''' % str(tuple(star_names))

        job = Gaia.launch_job_async(query, dump_to_file=True)

        # Your astropy table with results
        r = job.get_results()

        # keys=['source_id', 'phot_bp_mean_flux','ra_pmdec_corr','ra_error','ra',
        # 'pmra_error','ecl_lon','designation','l','phot_rp_mean_mag',
        # 'parallax_pmdec_corr','ra_parallax_corr','pmdec_error',
        # 'phot_g_mean_mag','pmra','parallax','radial_velocity',
        # 'radial_velocity_error','ra_dec_corr','parallax_error',
        # 'dec_pmdec_corr','dec_error','pmdec','parallax_over_error','b',
        # 'ref_epoch','ra_pmra_corr','dec_parallax_corr','phot_bp_mean_mag',
        # 'dec','dec_pmra_corr','pmra_pmdec_corr','parallax_pmra_corr','bp_rp',
        # 'ecl_lat']
        keys = ['source_id', 'phot_bp_mean_flux', 'ra_error', 'ra',
                'phot_rp_mean_mag',
                'phot_g_mean_mag', 'phot_bp_mean_mag', 'bp_rp']

        r2 = r[keys]
        d = dict(zip(r2['source_id'], r2))

    # gagne_filename = '../data/gagne_bonafide_full_kinematics_with_best_radial' \
    #                  '_velocity.fits'
    # gt = Table.read(gagne_filename)

    # print("Incorporate overlooked rvs compiled from the literature")
    # insertLitRVs(gt, banyan_data)

    # print("Adopt approximate masses from spectral types")
    # masses = np.array(
    #     [getMassFromSpectralType(stype) for stype in gt['Spectral type']]
    # )
    # gt['approx_mass'] = masses

    # # explore the significance of Bayesian parallax conversion:
    # par_comp = np.vstack((
    #      1000/gt['parallax'],
    #      1000/gt['parallax'] - (1000/(gt['parallax'] + gt['parallax_error'])),
    #      gt['r_est'],
    #      -0.5 * (gt['r_lo'] - gt['r_hi']),
    #      1/gt['parallax_over_error'] * 100)).T

    # par_comp[np.where((par_comp[:, -1] > 10) & (par_comp[:, -1] < 20))]

    # print("convert astrometric values into lsr-centric cartesian coordinates")
    # nrows = len(gt['source_id'])
    # empty_col = np.array(nrows * [np.nan])
    # cart_col_names = ['X', 'Y', 'Z', 'U', 'V', 'W',
    #                   'dX', 'dY', 'dZ', 'dU', 'dV', 'dW',
    #                   'c_XY', 'c_XZ', 'c_XU', 'c_XV', 'c_XW',
    #                           'c_YZ', 'c_YU', 'c_YV', 'c_YW',
    #                                   'c_ZU', 'c_ZV', 'c_ZW',
    #                                           'c_UV', 'c_UW',
    #                                                   'c_VW']

    # # insert empty columns
    # for col_name in cart_col_names:
    #     gt[col_name] = empty_col

    # for row_ix, gt_row in enumerate(gt):
    #     dim = 6
    #     if row_ix%10 == 0:
    #         print("{:02.2f}% done".format(row_ix / float(nrows) * 100.))
    #     astr_mean, astr_cov = gc.convertRecToArray(gt_row)
    #     xyzuvw_mean = coord.convertAstrometryToLSRXYZUVW(astr_mean)
    #     xyzuvw_cov = tf.transform_cov(
    #         astr_cov,
    #         coord.convertAstrometryToLSRXYZUVW,
    #         astr_mean,
    #         dim=dim,
    #     )
    #     # fill in cartesian mean
    #     for col_ix, col_name in enumerate(cart_col_names[:6]):
    #         gt_row[col_name] = xyzuvw_mean[col_ix]
    #
    #
    #     # fill in standard deviations
    #     xyzuvw_stds = np.sqrt(xyzuvw_cov[np.diag_indices(dim)])
    #     for col_ix, col_name in enumerate(cart_col_names[6:12]):
    #         gt_row[col_name] = xyzuvw_stds[col_ix]
    #
    #     correl_matrix = xyzuvw_cov / xyzuvw_stds / xyzuvw_stds.reshape(6,1)
    #     # fill in correlations
    #     for col_ix, col_name in enumerate(cart_col_names[12:]):
    #         gt_row[col_name] = correl_matrix[
    #             np.triu_indices(dim,k=1)[0][col_ix],
    #             np.triu_indices(dim,k=1)[1][col_ix]
    #         ]
    #         # I think I can write above line as:
    #         # gt_row[col_name] = correl_matrix[np.triu_indices(dim,k=1)][col_ix]

    # try:
    #     gt.write('../data/gagne_2_bonafide_full_kinematics_with_lit_and_best_radial' \
    #                      '_velocity.fits')
    # except:
    #     gt.write('../data/save_by_the_bell.fits')
