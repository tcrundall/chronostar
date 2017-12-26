#!/usr/bin/env python -W ignore
"""
test_utils
-----------------------------

Tests for `utils` module
"""
from __future__ import division, print_function

import os.path
import sys
import tempfile
import unittest

sys.path.insert(0, '..')  # hacky way to get access to module

import numpy as np
from chronostar import utils

class UtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.synth_file = os.path.join(self.tempdir, 'synth_data.pkl')
        self.tb_file = os.path.join(self.tempdir, 'tb_data.pkl')

    def tearDown(self):
        try:
            os.remove(self.synth_file)
        except OSError:
            pass
        try:
            os.remove(self.tb_file)
        except OSError:
            pass
        os.rmdir(self.tempdir)

    def test_generate_cov(self):
        dX = 5
        dY = 5
        dZ = 5
        dV = 3
        Cxy = 0.0
        Cxz = 0.0
        Cyz = 0.0

        #                X,Y,Z,U,V,W,1/dX,1/dY,1/dZ,1/dV,Cxy,Cxz,Cyz,age
        pars = np.array([0, 0, 0, 0, 0, 0, 1 / dX, 1 / dY, 1 / dZ, 1 / dV, Cxy, Cxz, Cyz, 10])

        test_cov = np.array([
            [dX ** 2, Cxy * dX * dY, Cxz * dX * dZ, 0.0, 0.0, 0.0],
            [Cxy * dX * dY, dY ** 2, Cyz * dY * dZ, 0.0, 0.0, 0.0],
            [Cxz * dX * dZ, Cyz * dY * dZ, dZ ** 2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, dV ** 2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, dV ** 2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, dV ** 2],
        ])

        gf_cov = utils.generate_cov(pars)

        self.assertTrue(np.allclose(test_cov, test_cov))
        self.assertTrue(np.allclose(test_cov, gf_cov))

        # orig calculation
        orig_cov = np.eye(6)
        # Fill in correlations
        orig_cov[np.tril_indices(3, -1)] = pars[10:13]
        orig_cov[np.triu_indices(3, 1)] = pars[10:13]
        # Note that 'pars' stores the inverse of the standard deviation
        # Convert correlation to orig_covariance for position.
        for i in range(3):
            orig_cov[i, :3] *= 1 / pars[6:9]
            orig_cov[:3, i] *= 1 / pars[6:9]
        # Convert correlation to orig_covariance for velocity.
        for i in range(3, 6):
            orig_cov[i, 3:] *= 1 / pars[9]
            orig_cov[3:, i] *= 1 / pars[9]
        # Generate inverse cov manp.trix and its determinant
        self.assertTrue(np.allclose(orig_cov, gf_cov))

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(UtilsTestCase)
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=1).run(suite())
