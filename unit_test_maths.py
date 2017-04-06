#! /usr/bin/env python
"""
    Small unit test used to confirm two different derivations of the
    overlap integral of two multivariate gaussians are equivalent.
    
    Appears to be equivalent within 7-8 sig figs
"""

import chronostar
from chronostar.fit_group import compute_overlap as co
import numpy as np
#from math import isclose
import pdb

def new_co(A_cov,a_mn,B_cov,b_mn):
    """
    This is an alternative derivation of the overlap integral between
    two multivariate gaussians. This is *not* the version implemented
    in the swigged C module.

    Compute the overlap integral between a star and group mean + covariance matrix
    in six dimensions, including some temporary variables for speed and to match the 
    notes.
    
    This is the first function to be converted to a C program in order to speed up.
    """

    AcpBc = (A_cov + B_cov)
    AcpBc_det = np.linalg.det(AcpBc)

    AcpBc_i = np.linalg.inv(AcpBc)
    #amn_m_bmn = a_mn - b_mn

    overlap = np.exp(-0.5 * (np.dot(a_mn-b_mn, np.dot(AcpBc_i,a_mn-b_mn) )) )
    overlap *= 1.0/((2*np.pi)**3.0 * np.sqrt(AcpBc_det))
    return overlap

star_params = chronostar.fit_group.read_stars("results/bp_TGAS2_traceback_save.pkl")

icov = star_params["xyzuvw_icov"]
cov = star_params["xyzuvw_cov"]
mean = star_params["xyzuvw"]
det = star_params["xyzuvw_icov_det"]

nstars = 407
threshold = 1e-9

for i in range(0,nstars):
    A_cov = cov[i,0]
    A_icov = icov[i,0]
    a_mn = mean[i,0]
    A_idet = det[i,0]
    
    B_cov = cov[i,1]
    B_icov = icov[i,1]
    b_mn = mean[i,1]
    B_idet = det[i,1]

    mikes_ol = co(A_icov,a_mn,A_idet,B_icov,b_mn,B_idet)
    tims_ol = new_co(A_cov,a_mn,B_cov,b_mn)

assert(np.all( (mikes_ol - tims_ol)/mikes_ol < threshold)),\
       "Not all {} star overlaps are consistent within {}".format(nstars,
                                                                  threshold)
print("___ check_maths.py passes all tests ___")
