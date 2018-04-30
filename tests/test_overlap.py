#! /usr/bin/env python
"""
    Small unit test used to confirm two different derivations of the
    overlap integral of two multivariate gaussians are equivalent.
    
    Appears to be equivalent within 7-8 sig figs
"""
import numpy as np
#from math import isclose
import sys
sys.path.insert(0, '..')

import chronostar.groupfitter as gf
from chronostar.retired.fit_group import compute_overlap as co
from chronostar.retired.fit_group import read_stars


def compute_overlap(A, a, A_det, B, b, B_det):
    """Compute the overlap integral between a star and group mean +
    covariance matrix in six dimensions, including some temporary variables
    for speed and to match the notes.

    This is the first function to be converted to a C program in order to
    speed up.
    """

    # Preliminaries - add matrices together. This might make code more
    # readable?
    # Or might not.
    ApB = A + B
    AapBb = np.dot(A, a) + np.dot(B, b)

    # Compute determinants.
    ApB_det = np.linalg.det(ApB)

    # Error checking (not needed in C once shown to work?) This shouldn't
    # ever happen, as
    # the determinants of the sum of positive definite matrices is
    # greater than the sum of their determinants
    if (ApB_det < 0) | (B_det < 0):
        pdb.set_trace()
        return -np.inf

    # Solve for c
    c = np.linalg.solve(ApB, AapBb)

    # Compute the overlap formula.
    overlap = np.exp(-0.5 * (np.dot(b - c, np.dot(B, b - c)) + \
                             np.dot(a - c, np.dot(A, a - c))))
    overlap *= np.sqrt(B_det * A_det / ApB_det) / (2 * np.pi) ** 3.0

    return overlap

def new_co(A_cov,a_mn,B_cov,b_mn):
    """
    This is an alternative derivation of the overlap integral between
    two multivariate gaussians. This is *not* the version implemented
    in the swigged C module.

    Compute the overlap integral between a star and group mean + covariance
    matrix in six dimensions, including some temporary variables for speed and
    to match the notes.
    
    This is the first function to be converted to a C program in order to
    speed up.
    """

    AcpBc = (A_cov + B_cov)
    AcpBc_det = np.linalg.det(AcpBc)

    AcpBc_i = np.linalg.inv(AcpBc)
    #amn_m_bmn = a_mn - b_mn

    overlap = np.exp(-0.5 * (np.dot(a_mn-b_mn, np.dot(AcpBc_i,a_mn-b_mn) )) )
    overlap *= 1.0/((2*np.pi)**3.0 * np.sqrt(AcpBc_det))
    return overlap

def test_maths_friendl():
    """
    Gonna try and run various implementations on very friendly gaussians.
    :return:
    """
    pass

def test_maths_broken():
    """
    Currently only returns -inf... warrants investigation
    """
    xyzuvw_file = "../data/fed_stars_20_xyzuvw.fits"
    xyzuvw_dict = gf.loadXYZUVW(xyzuvw_file)

    mean = xyzuvw_dict['xyzuvw']
    cov = xyzuvw_dict['xyzuvw_cov']
    nstars = cov.shape[0]

    tims_old_old_lnol = np.zeros(nstars-1)
    A_cov = cov[0]
    a_mn = mean[0]

    for i in range(0,nstars - 1):
        B_cov = cov[i+1]
        b_mn = mean[i+1]

        tims_old_old_lnol[i] = np.log(new_co(A_cov,a_mn,B_cov,b_mn))

    tims_old_lnol = gf.get_lnoverlaps(A_cov, a_mn, cov[1:], mean[1:], nstars-1)
    tims_new_lnol = gf.slowGetLogOverlaps(A_cov, a_mn, cov[1:], mean[1:], nstars - 1)
    import pdb;pdb.set_trace()
    assert np.allclose(tims_new_lnol, tims_old_lnol, rtol=1e-2)
    assert np.allclose(tims_new_lnol, tims_old_old_lnol, rtol=1e-2)
