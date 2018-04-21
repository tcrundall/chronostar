#!/usr/bin/env python -W ignore
"""
Checks the various implemntations of the overlap integral
for correctness by comparing values against eachother
"""

import sys
sys.path.insert(0,'..') #hacky way to get access to module
import numpy as np
import unittest

import chronostar
from chronostar.retired.fit_group import compute_overlap as co
from chronostar._overlap import get_lnoverlaps as swig_clnos


class MathsTestCase(unittest.TestCase):
    def new_clno(self,A_cov,a_mn,B_cov,b_mn,debug=False):
        """
        This is an alternative derivation of the overlap integral between
        two multivariate gaussians. This is *not* the version implemented
        in the swigged C module.

        Compute the overlap integral between a star and group mean + covariance
        matrix in six dimensions, including some temporary variables for speed
        and to match the notes.
        """
        BpA = (B_cov + A_cov)
        BpA_det = np.linalg.det(BpA)

        BpA_i = np.linalg.inv(BpA)
        bma = b_mn - a_mn

        bma_BpAi_bma = np.dot(b_mn-a_mn,np.dot(BpA_i,b_mn-a_mn) )

        ln_overlap = 6*np.log(2*np.pi)
        ln_overlap += np.log(BpA_det)
        if debug:
            print("Printing BpA:\n{}".format(BpA))
            print("Printing bma:\n{}".format(bma))
            print("Printing BpA_i:\n{}".format(BpA_i))
            print("ln(det(BpA)) added:\n{}\n".format(np.log(BpA_det)))
            print("result so far:\n{}".format(ln_overlap))
            print("bma_BpAi_bma:\n{}".format(bma_BpAi_bma))

        # IMPLEMENTATION PAUSED
        #ln_overlap += np.log(Bp

        ln_overlap += bma_BpAi_bma
        if debug:
            print("result after bma_BpAi_bma:\n{}".format(ln_overlap))

        ln_overlap *= -0.5

        #ln_overlap = -0.5*(np.log( (2*np.pi)**6 * BpA_det) + bma_BpAi_bma)
        if debug:
            print("Final result:\n{}".format(ln_overlap))
        return ln_overlap

    def new_co(self,A_cov,a_mn,B_cov,b_mn,debug=False):
        """
        This is an alternative derivation of the overlap integral between
        two multivariate gaussians. This is *not* the version implemented
        in the swigged C module.

        Compute the overlap integral between a star and group mean + covariance
        matrix in six dimensions, including some temporary variables for speed
        and to match the notes.
        """
        BpA = (B_cov + A_cov)
        BpA_det = np.linalg.det(BpA)

        BpA_i = np.linalg.inv(BpA)
        #amn_m_bmn = a_mn - b_mn

        bma = b_mn - a_mn

        bma_BpAi_bma = np.dot(a_mn-b_mn,np.dot(BpA_i,a_mn-b_mn) )
        if debug:
            print("BpA:\n{}".format(BpA))
            print("BpA_det: {}".format(BpA_det))
            print("BpA_i:\n{}".format(BpA_i))
            print("bma: {}".format(bma))
            print("ln_BpA_det:\n{}".format(np.log(BpA_det)))
            print("bma_BpAi_bma:\n{}".format(bma_BpAi_bma))

        overlap = np.exp(-0.5*(np.dot(a_mn-b_mn,np.dot(BpA_i,a_mn-b_mn) )) )
        overlap *= 1.0/((2*np.pi)**3.0 * np.sqrt(BpA_det))
        return overlap

    @unittest.skip("for low level debugging")
    def test_swig_verbose(self):
        star_params = chronostar.retired.fit_group.read_stars(
            "../data/bp_TGAS2_traceback_save.pkl")

        nstars = 2
        #nstars = mean.shape[0]

        icov = star_params["xyzuvw_icov"][0:nstars,0]
        cov = star_params["xyzuvw_cov"][0:nstars,0]
        mean = star_params["xyzuvw"][0:nstars,0]
        det = star_params["xyzuvw_icov_det"][0:nstars,0]

        gr_cov  = cov[0]
        gr_icov = icov[0]
        gr_mn   = mean[0]
        gr_icov_det = np.linalg.det(gr_icov)

        # Confirm that calculating overlap is the same as 
        # e^ln_overlap
        self.assertTrue(
            np.allclose(
                [self.new_co(gr_cov,gr_mn,cov[0],mean[0])],
                [np.exp(self.new_clno(gr_cov,gr_mn,cov[0],mean[0]))]
            )
        )

        self.assertTrue(
            np.allclose(
                [self.new_co(gr_cov,gr_mn,cov[1],mean[1])],
                [np.exp(self.new_clno(gr_cov,gr_mn,cov[1],mean[1]))]
            )
        )

        print("-----------------------------\n"\
              "--         python          --\n"\
              "-----------------------------\n")

        tims_ol1 = self.new_clno(gr_cov,gr_mn,cov[0],mean[0],debug=True)
        tims_ol2 = self.new_clno(gr_cov,gr_mn,cov[1],mean[1],debug=True)

        print("Results:\n{}\n{}".format(tims_ol1, tims_ol2))

        print("-----------------------------\n"\
              "--            C            --\n"\
              "-----------------------------\n")

        swig_ols = np.exp(
            swig_clnos(
                gr_cov, gr_mn,
                cov[:], mean[:],
                nstars,
            )
        )
        print(np.exp(swig_ols))

    def test_overlap(self):
        star_params = chronostar.retired.fit_group.read_stars(
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

        swig_lnols = swig_clnos(
            gr_cov, gr_mn,
            cov[:,0], mean[:,0],
            nstars,
        )
        
        swig_ols = np.exp(swig_lnols)

        mikes_ols = np.zeros(nstars)
        tims_ols = np.zeros(nstars)
        tims_lnols = np.zeros(nstars)

        # set values
        for i in range(0,nstars):
            B_cov = cov[i,0]
            B_icov = icov[i,0]
            b_mn = mean[i,0]
            B_idet = det[i,0]

            mikes_ols[i] = co(gr_icov,gr_mn,gr_icov_det,B_icov,b_mn,B_idet)
            tims_ols[i] = self.new_co(gr_cov,gr_mn,B_cov,b_mn)
            tims_lnols[i] = self.new_clno(gr_cov,gr_mn,B_cov,b_mn)

        #pdb.set_trace()

        # check values
        for i in range(nstars):
            # formatted this way allows ol values to both be 0.0
            self.assertTrue(( mikes_ols[i] - tims_ols[i]) <=\
                mikes_ols[i]*threshold1,
                "{}: We have {} and {}".format(i, mikes_ols[i], tims_ols[i])
            )

            self.assertTrue(( abs(tims_lnols[i] - swig_lnols[i])) <=\
                abs(tims_lnols[i])*threshold1,
                "{}: We have {} and {}".format(i, tims_lnols[i], swig_lnols[i])
            )

            self.assertTrue(( mikes_ols[i] - swig_ols[i]) <=\
                mikes_ols[i]*threshold1,
                "{}: We have {} and {}".\
                format(i, mikes_ols[i], swig_ols[i])
            )
    
def suite():
    tests = ['test_overlap', 'test_swig_verbose']
    return unittest.TestSuite(map(MathsTestCase, tests))

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

sys.path.insert(0,'.') #hacky way to get access to module
