from __future__ import print_function, division

import numpy as np
from astropy.table import Table

CART_COL_NAMES = ['X', 'Y', 'Z', 'U', 'V', 'W',
                  'dX', 'dY', 'dZ', 'dU', 'dV', 'dW',
                  'c_XY', 'c_XZ', 'c_XU', 'c_XV', 'c_XW',
                  'c_YZ', 'c_YU', 'c_YV', 'c_YW',
                  'c_ZU', 'c_ZV', 'c_ZW',
                  'c_UV', 'c_UW',
                  'c_VW']

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
    # CART_COL_NAMES = ['X', 'Y', 'Z', 'U', 'V', 'W',
    #                   'dX', 'dY',   'dZ',   'dU',   'dV',   'dW',
    #                         'c_XY', 'c_XZ', 'c_XU', 'c_XV', 'c_XW',
    #                                 'c_YZ', 'c_YU', 'c_YV', 'c_YW',
    #                                         'c_ZU', 'c_ZV', 'c_ZW',
    #                                                 'c_UV', 'c_UW',
    #                                                         'c_VW']
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

def insertCartesainData(gt_row, xyzuvw_mean, xyzuvw_cov):
    dim = 6
    # CART_COL_NAMES = ['X', 'Y', 'Z', 'U', 'V', 'W',
    #                   'dX', 'dY', 'dZ', 'dU', 'dV', 'dW',
    #                   'c_XY', 'c_XZ', 'c_XU', 'c_XV', 'c_XW',
    #                           'c_YZ', 'c_YU', 'c_YV', 'c_YW',
    #                                   'c_ZU', 'c_ZV', 'c_ZW',
    #                                           'c_UV', 'c_UW',
    #                                                   'c_VW']

    # fill in cartesian mean
    try:
        for col_ix, col_name in enumerate(CART_COL_NAMES[:6]):
            gt_row[col_name] = xyzuvw_mean[col_ix]
    except IndexError:
        import pdb; pdb.set_trace()

    # fill in standard deviations
    xyzuvw_stds = np.sqrt(xyzuvw_cov[np.diag_indices(dim)])
    for col_ix, col_name in enumerate(CART_COL_NAMES[6:12]):
        gt_row[col_name] = xyzuvw_stds[col_ix]

    correl_matrix = xyzuvw_cov / xyzuvw_stds / xyzuvw_stds.reshape(6, 1)
    # fill in correlations
    for col_ix, col_name in enumerate(CART_COL_NAMES[12:]):
        gt_row[col_name] = correl_matrix[
            np.triu_indices(dim, k=1)[0][col_ix],
            np.triu_indices(dim, k=1)[1][col_ix]
        ]

def eraseCartesianData(gt_row):
    """
    Replace all cartesian data with blanks (for erasing binary companions that
    have been incoroporated into host star's kinematics)
    """
    dim = 6

    for col_name in CART_COL_NAMES:
        gt_row[col_name] = np.nan


if __name__ == '__main__':
    gagne_filename = '../data/gagne_bonafide_full_kinematics_with_lit' \
                     '_and_best_radial_velocity.fits'
    gt = Table.read(gagne_filename)
    new_gagne_filename = '../data/gagne_bonafide_full_kinematics_with_lit' \
                     '_and_best_radial_velocity_comb_binars.fits'

    # companion_ixs = np.where(gt['Companion'] == 'True')
    # host_ixs = (companion_ixs[0]-1,)

    row_ix = 0
    while row_ix < len(gt['Companion']) - 1:
        # find next instance of host
        system_ixs = []
        while (row_ix+1 < len(gt['Companion']) and
               gt[row_ix+1]['Companion'] == 'False'):
            row_ix += 1
        # Check if we've reached end of table
        if row_ix == len(gt['Companion'])-1:
            break
        system_ixs.append(row_ix)

        # find subsequent associated companions
        row_ix += 1
        while row_ix < len(gt['Companion']) and gt[row_ix]['Companion'] == 'True':
            system_ixs.append(row_ix)
            row_ix += 1

        print(system_ixs)

        masses = []
        means = []
        cov_mats = []
        for row_ix in system_ixs:
            # Check 'U' (for e.g.) has been calculated
            # (requires both RV and parallax so good guide for 6D completeness)
            if np.isfinite(gt[row_ix]['U']):
                masses.append(gt[row_ix]['approx_mass'])
                mean, cov_mat = buildMeanAndCovMatFromRow(gt[row_ix])
                means.append(mean)
                cov_mats.append(cov_mat)

        masses = np.array(masses)
        means = np.array(means)
        cov_mats = np.array(cov_mats)

        nstars = len(means)
        if nstars != 0:
            # establish average of available masses
            av_present_mass = masses[np.where(np.isfinite(masses))].mean()
            # replace everything with 1 if there are no valid masses
            if np.isnan(av_present_mass):
                masses[:] = 1.0
            # replace any missing masses with average of available masses
            else:
                masses[np.where(np.isnan(masses))] = av_present_mass

            assert np.all(np.isfinite(masses))
            total_mass = np.sum(masses)

            aver_mean = (means.T * masses).sum(axis=-1) / total_mass
            aver_cov_mat = (cov_mats.T * masses).sum(axis=-1) / total_mass

            insertCartesainData(gt[system_ixs[0]], aver_mean, aver_cov_mat)

            for system_ix in system_ixs[1:]:
                eraseCartesianData(gt[system_ix])

    for pos_dim in CART_COL_NAMES[:3]:
        gt[pos_dim].unit = 'pc'
    for vel_dim in CART_COL_NAMES[3:6]:
        gt[vel_dim].unit = 'km/s'
    for pos_std in CART_COL_NAMES[6:9]:
        gt[pos_std].unit = 'pc'
    for vel_std in CART_COL_NAMES[9:12]:
        gt[vel_std].unit = 'km/s'
    gt['approx_mass'].unit = 'M_solar'

    gt.write(new_gagne_filename)



