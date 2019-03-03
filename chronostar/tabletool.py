"""
A bunch of functions that help handle stellar data stored as
astropy table.
"""

import numpy as np
from astropy.table import Table


def convertTableAstroToXYZUVW(table):
    pass


def buildXYZUVWCovMatrices(table):
    corr_col_names = ['c_XY', 'c_XZ', 'c_XU', 'c_XV', 'c_XW',
                              'c_YZ', 'c_YU', 'c_YV', 'c_YW',
                                      'c_ZU', 'c_ZV', 'c_ZW',
                                              'c_UV', 'c_UW',
                                                      'c_VW']

    nstars = len(table)
    standard_devs = np.vstack((table['dX'], table['dY'], table['dZ'],
                               table['dU'], table['dV'], table['dW'])).T
    xyzuvw_base_cov = np.array(nstars * [np.eye(6)])
    indices = np.triu_indices(6, 1)
    for ix in range(15):
        fst_ix = indices[0][ix]
        snd_ix = indices[1][ix]
        xyzuvw_base_cov[:, fst_ix, snd_ix] = table[corr_col_names[ix]]
        xyzuvw_base_cov[:, snd_ix, fst_ix] = table[corr_col_names[ix]]

    xyzuvw_base_cov = np.einsum('ijk,ij->ijk', xyzuvw_base_cov, standard_devs)
    xyzuvw_base_cov = np.einsum('ijk,ik->ijk', xyzuvw_base_cov, standard_devs)
    return xyzuvw_base_cov

def convertTableXYZUVWToArray(table, row_ix=None, nrows=None):
    if isinstance(table, str):
        table = Table.read(table)

    xyzuvw_mean = np.vstack((table['X'], table['Y'], table['Z'],
                             table['U'], table['V'], table['W'])).T

    xyzuvw_covs = buildXYZUVWCovMatrices(table)
    return xyzuvw_mean, xyzuvw_covs
