#!/usr/bin/env python
"""
Checks the various implemntations of the overlap integral
for correctness by comparing values against eachother
"""

import sys
sys.path.insert(0,'..') #hacky way to get access to module
import numpy as np
import pdb
import unittest

import chronostar
from chronostar.fit_group import compute_overlap as co
from chronostar._overlap import get_overlap as swig_co
from chronostar._overlap import get_overlaps as swig_cos


class TestStringMethods(unittest.TestCase):
    def new_co(self,A_cov,a_mn,B_cov,b_mn):
        """
        This is an alternative derivation of the overlap integral between
        two multivariate gaussians. This is *not* the version implemented
        in the swigged C module.

        Compute the overlap integral between a star and group mean + covariance
        matrix in six dimensions, including some temporary variables for speed
        and to match the notes.
        """
        AcpBc = (A_cov + B_cov)
        AcpBc_det = np.linalg.det(AcpBc)

        AcpBc_i = np.linalg.inv(AcpBc)
        #amn_m_bmn = a_mn - b_mn

        overlap = np.exp(-0.5*(np.dot(a_mn-b_mn,np.dot(AcpBc_i,a_mn-b_mn) )) )
        overlap *= 1.0/((2*np.pi)**3.0 * np.sqrt(AcpBc_det))
        return overlap

    def test_overlap(self):
        star_params = chronostar.fit_group.read_stars(
            "../data/bp_TGAS2_traceback_save.pkl")

        icov = star_params["xyzuvw_icov"]
        cov = star_params["xyzuvw_cov"]
        mean = star_params["xyzuvw"]
        det = star_params["xyzuvw_icov_det"]

        nstars = mean.shape[0]
        threshold1 = 1e-8
        threshold2 = 1e-3
        threshold3 = 1e-4

        # take the "group" as the first star
        # take all stars at time 0
        gr_cov  = cov[0,0]
        gr_icov = icov[0,0]
        gr_mn   = mean[0,0]
        gr_icov_det = np.linalg.det(gr_icov)
        
        swig_a_ols = swig_cos(
            gr_icov, gr_mn, gr_icov_det,
            icov[:,0], mean[:,0], det[:,0],
            nstars,
            )

        for i in range(0,nstars):
            B_cov = cov[i,0]
            B_icov = icov[i,0]
            b_mn = mean[i,0]
            B_idet = det[i,0]

            mikes_ol = co(gr_icov,gr_mn,gr_icov_det,B_icov,b_mn,B_idet)
            tims_ol = self.new_co(gr_cov,gr_mn,B_cov,b_mn)
            swig_s_ol = swig_co(
                gr_icov, gr_mn, np.linalg.det(gr_icov), B_icov, b_mn,
                np.linalg.det(B_icov)
                )

            # formatted this way allows ol values to both be 0.0
            self.assertTrue(( mikes_ol - tims_ol) <=\
                mikes_ol*threshold1,
                "{}: We have {} and {}".format(i, mikes_ol, tims_ol))

            self.assertTrue(( mikes_ol - swig_s_ol) <=\
                mikes_ol*threshold2,
                "{}: We have {} and {}".format(i, mikes_ol, swig_s_ol))

            self.assertTrue(( mikes_ol - swig_a_ols[i]) <=\
                mikes_ol*threshold3,
                "{}: We have {} and {}".format(i, mikes_ol, swig_a_ols[i]))

if __name__ == '__main__':
    unittest.main()

sys.path.insert(0,'.') #hacky way to get access to module
