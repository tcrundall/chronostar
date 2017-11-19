import unittest

import sys
sys.path.insert(0,'..') #hacky way to get access to module
import numpy as np
import pdb

import chronostar
from chronostar.fit_group import compute_overlap as co
from chronostar._overlap import get_overlap as swig_co
from chronostar._overlap import get_overlaps as swig_cos


class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

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

        mikes_ols = np.zeros(nstars)
        tims_ols  = np.zeros(nstars)
        swig_s_ols = np.zeros(nstars)
        swig_a_ols = np.zeros(nstars)
    
        # take the "group" as the first star
        # take all stars at time 0
        swig_a_ols = swig_cos(
            icov[0,0], mean[0,0],
            np.linalg.det(icov[0,0]),
            icov[:,0], mean[:,0],
            det[:,0],
            nstars,
            )

        for i in range(0,nstars):
            A_cov = cov[i,0]
            A_icov = icov[i,0]
            a_mn = mean[i,0]
            A_idet = det[i,0]

            A_cov = cov[0,0]
            A_icov = icov[0,0]
            a_mn = mean[0,0]
            A_idet = det[0,0]

            B_cov = cov[i,0]
            B_icov = icov[i,0]
            b_mn = mean[i,0]
            B_idet = det[i,0]

            mikes_ols[i] = co(A_icov,a_mn,A_idet,B_icov,b_mn,B_idet)
            tims_ols[i] = self.new_co(A_cov,a_mn,B_cov,b_mn)
            swig_s_ols[i] = swig_co(A_icov,a_mn,
                                         np.linalg.det(A_icov),
                                         B_icov,b_mn,
                                         np.linalg.det(B_icov)
                                         )

            # formatted this way allows ol values to both be 0.0
            self.assertTrue(( mikes_ols[i] - tims_ols[i]) <=\
                mikes_ols[i]*threshold1,
                "{}: We have {} and {}".format(i, mikes_ols[i], tims_ols[i]))

            self.assertTrue(( mikes_ols[i] - swig_s_ols[i]) <=\
                mikes_ols[i]*threshold2,
                "{}: We have {} and {}".format(i, mikes_ols[i], swig_s_ols[i]))

            self.assertTrue(( mikes_ols[i] - swig_a_ols[i]) <=\
                mikes_ols[i]*threshold2,
                "{}: We have {} and {}".format(i, mikes_ols[i], swig_a_ols[i]))

        self.assertTrue(np.all( abs(mikes_ols - tims_ols) <=\
                                abs(mikes_ols*threshold1)),\
               "Not all {} star overlaps are consistent within {}".\
                format(nstars, threshold1))

if __name__ == '__main__':
    unittest.main()
