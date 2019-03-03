#! /usr/bin/env python
"""
    Small unit test used to confirm two different derivations of the
    overlap integral of two multivariate gaussians are equivalent.
    
    Appears to be equivalent within 7-8 sig figs
"""
import logging
import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.groupfitter as gf
from chronostar.likelihood import slowGetLogOverlaps as p_lno
from chronostar._overlap import get_lnoverlaps as c_lno


def co2(A, a, B, b):
    """
    This is an alternative derivation of the overlap integral between
    two multivariate gaussians. This is the version implemented
    in the swigged C module.

    Parameters
    ----------
    A : (n x n) np.float array
        Covariance matrix of first Gaussian distribution
    a : (n) np.float array
        Mean of first Gaussian distribution
    B : (n x n) np.float array
        Covariance matrix of second Gaussian distribution
    b : (n) np.float array
        Mean of second Gaussian distribution
    """
    ApB = (A + B)
    ApB_det = np.linalg.det(ApB)
    ApB_i = np.linalg.inv(ApB)
    overlap = np.exp(-0.5 * (np.dot(a - b, np.dot(ApB_i, a - b))))
    overlap *= 1.0 / ((2 * np.pi) ** 3.0 * np.sqrt(ApB_det) )
    return overlap


def co1(A_cov, a, B_cov, b):
    """
    The original python function written by Mike yeaaaaarrss ago

    Parameters
    ----------
    A_cov : (n x n) np.float array
        Covariance matrix of first Gaussian distribution
    a : (n) np.float array
        Mean of first Gaussian distribution
    B_cov : (n x n) np.float array
        Covariance matrix of second Gaussian distribution
    b : (n) np.float array
        Mean of second Gaussian distribution
    """
    A = np.linalg.inv(A_cov)
    B = np.linalg.inv(B_cov)
    A_det = np.linalg.det(A)
    B_det = np.linalg.det(B)

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
        return -np.inf

    # Solve for c
    c = np.linalg.solve(ApB, AapBb)

    # Compute the overlap formula.
    overlap = np.exp(-0.5 * (np.dot(b - c, np.dot(B, b - c)) + \
                             np.dot(a - c, np.dot(A, a - c))))
    overlap *= np.sqrt(B_det * A_det / ApB_det) / (2 * np.pi) ** 3.0
    return overlap


def test_pythonFuncs():
    """
    TODO: remove the requirements of file, have data stored in file?
    """
    try:
        xyzuvw_file = "../data/fed_stars_20_xyzuvw.fits"
        xyzuvw_dict = gf.loadXYZUVW(xyzuvw_file)
    except IOError:
        print("Required data file missing")
        assert False

    star_means = xyzuvw_dict['xyzuvw']
    star_covs = xyzuvw_dict['xyzuvw_cov']
    nstars = star_means.shape[0]

    group_mean = np.mean(star_means, axis=0)
    group_cov = np.cov(star_means.T)

    co1s = []
    co2s = []
    for i, (scov, smn) in enumerate(zip(star_covs, star_means)):
        print(i)
        co1s.append(co1(group_cov, group_mean, scov, smn))
        co2s.append(co2(group_cov, group_mean, scov, smn))
    co1s = np.array(co1s)
    co2s = np.array(co2s)
    co3s = np.exp(p_lno(group_cov, group_mean, star_covs, star_means))
    assert np.allclose(co1s, co2s)
    assert np.allclose(co2s, co3s)
    assert np.allclose(co1s, co3s)

    # note that most overlaps go to 0, but the log overlaps retains the
    # information
    co1s = []
    co2s = []
    for i, (scov, smn) in enumerate(zip(star_covs, star_means)):
        co1s.append(co1(star_covs[15], star_means[15], scov, smn))
        co2s.append(co2(star_covs[15], star_means[15], scov, smn))
    co1s = np.array(co1s)
    co2s = np.array(co2s)
    lnos = p_lno(star_covs[15], star_means[15], star_covs, star_means)
    co3s = np.exp(lnos)
    assert np.allclose(co1s, co2s)
    assert np.allclose(co2s, co3s)
    assert np.allclose(co1s, co3s)


def test_swigImplementation():
    """
    Compares the swigged c implementation against the python one in groupfitter
    """
    xyzuvw_file = "../data/fed_stars_20_xyzuvw.fits"
    xyzuvw_dict = gf.loadXYZUVW(xyzuvw_file)

    star_means = xyzuvw_dict['xyzuvw']
    star_covs = xyzuvw_dict['xyzuvw_cov']
    nstars = star_means.shape[0]

    gmn = np.mean(star_means, axis=0)
    gcov = np.cov(star_means.T)

    p_lnos = p_lno(gcov, gmn, star_covs, star_means)
    c_lnos = c_lno(gcov, gmn, star_covs, star_means, nstars)

    assert np.allclose(p_lnos, c_lnos)
    assert np.isfinite(p_lnos).all()
    assert np.isfinite(c_lnos).all()
