"""
author: Marusa Zerjal 2019 - 07 - 29

Determine background overlaps using means and covariances for both
background and stars.
Covariance matrices for the background are Identity*bandwidth.

Parameters
----------
background_means: [nstars,6] float array_like
    Phase-space positions of some star set that greatly envelops points
    in question. Typically contents of gaia_xyzuvw.npy, or the output of
    >> tabletool.build_data_dict_from_table(
               '../data/gaia_cartesian_full_6d_table.fits',
                historical=True)['means']
star_means: [npoints,6] float array_like
    Phase-space positions of stellar data that we are fitting components to
star_covs: [npoints,6,6] float array_like
    Phase-space covariances of stellar data that we are fitting components to

Output is a file with ln_bg_ols. Same order as input datafile.
No return.

bg_lnols: [nstars] float array_like
    Background log overlaps of stars with background probability density
    function.

Notes
-----
We invert the vertical values (Z and U) because the typical background
density should be symmetric along the vertical axis, and this distances
stars from their siblings. I.e. association stars aren't assigned
higher background overlaps by virtue of being an association star.

Edits
-----
TC 2019-05-28: changed signature such that it follows similar usage as
               get_kernel_densitites
"""

from __future__ import print_function, division

import numpy as np
from mpi4py import MPI

import logging

# The placement of logsumexp varies wildly between scipy versions
import scipy
_SCIPY_VERSION= [int(v.split('rc')[0])
                 for v in scipy.__version__.split('.')]
if _SCIPY_VERSION[0] == 0 and _SCIPY_VERSION[1] < 10:
    from scipy.maxentropy import logsumexp
elif ((_SCIPY_VERSION[0] == 1 and _SCIPY_VERSION[1] >= 3) or
    _SCIPY_VERSION[0] > 1):
    from scipy.special import logsumexp
else:
    from scipy.misc import logsumexp


import sys
sys.path.insert(0, '..')

try:
    print('Using C implementation in expectmax')
    from _overlap import get_lnoverlaps
except:
    print("WARNING: Couldn't import C implementation, using slow pythonic overlap instead")
    logging.info("WARNING: Couldn't import C implementation, using slow pythonic overlap instead")
    from chronostar.likelihood import slow_get_lnoverlaps as get_lnoverlaps


def log_message(msg, symbol='.', surround=False):
    """Little formatting helper"""
    res = '{}{:^40}{}'.format(5*symbol, msg, 5*symbol)
    if surround:
        res = '\n{}\n{}\n{}'.format(50*symbol, res, 50*symbol)
    logging.info(res)


comm = MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()


if rank == 0:
    # PREPARE STELLAR DATA
    datafile = 'data_table_cartesian_100k.fits' # SHOULD BE CARTESIAN
    data_table = tabletool.read(datafile)
    historical = 'c_XU' in data_table.colnames
    data_table = data_table[:20] # for testing
    print('DATA_TABLE READ', len(data_table))

    print('data_dict')
    data_dict = tabletool.build_data_dict_from_table(
        data_table,
        get_background_overlaps=False, # bg overlap not available yet
        historical=historical,
    )
    star_means = data_dict['means']
    star_covs = data_dict['covs']

    # PREPARE BACKGROUND DATA
    print('Read background Gaia data')
    background_means = tabletool.build_data_dict_from_table(
        '/home/tcrun/chronostar/data/gaia_cartesian_full_6d_table.fits',
        only_means=True,
    )

    # Inverting the vertical values
    star_means = np.copy(star_means)
    star_means[:, 2] *= -1
    star_means[:, 5] *= -1

    # Background covs with bandwidth using Scott's rule
    d = 6.0 # number of dimensions
    nstars = background_means.shape[0]
    bandwidth = nstars**(-1.0 / (d + 4.0))
    background_cov = np.cov(background_means.T) * bandwidth ** 2
    background_covs = np.array(nstars * [background_cov]) # same cov for every star


    # SPLIT DATA into multiple processes
    indices_chunks = np.array_split(range(nstars), size)
    star_means = [star_means[i] for i in indices_chunks]
    star_covs = [star_covs[i] for i in indices_chunks]


# BROADCAST CONSTANTS
nstars = comm.bcast(nstars, root=0)
background_means = comm.bcast(background_means, root=0)
background_covs = comm.bcast(background_covs, root=0)

# SCATTER DATA
star_means = comm.scatter(star_means, root=0)
star_covs = comm.scatter(star_covs, root=0)



# EVERY PROCESS DOES THIS FOR ITS DATA
bg_ln_ols=[]
for star_cov, star_mean in zip(star_covs, star_means):
    try:
        bg_lnol = get_lnoverlaps(star_cov, star_mean, background_covs,
                                 background_means, nstars)
        bg_lnol = logsumexp(bg_lnol)  # sum in linear space
    except:
        # TC: Changed sign to negative (surely if it fails, we want it to
        # have a neglible background overlap?
        print('bg ln overlap failed, setting it to -inf')
        bg_lnol = -np.inf

    bg_ln_ols.append(bg_lnol)



# GATHER DATA
bg_ln_ols_result = comm.gather(bg_ln_ols, root=0)
if rank == 0:
    print  ('master collected: ', bg_ln_ols_result)

    np.savetxt('bgols_multi_testing.dat')