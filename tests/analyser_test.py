#!/usr/bin/env python -W ignore
"""
analyser_test.py
----------------------------

Tests for `analyser` module.

To Do:
"""

import os.path
import sys
import tempfile
import unittest

sys.path.insert(0, '..')  # hacky way to get access to module

import numpy as np
import chronostar.analyser as an
import pdb
import pickle


class AnalyserTestCase(unittest.TestCase):
    def setUp(self):

        self.tempdir = tempfile.mkdtemp()
        self.synth_file = os.path.join(self.tempdir, 'synth_data.pkl')
        self.tb_file = os.path.join(self.tempdir, 'tb_data.pkl')

        mock_twa_pars = [
            -80, 80, 50, 10, -20, -5, 5, 5, 5, 2, 0.0, 0.0, 0.0, 7, 40
        ]

        NGROUPS = 1

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

    def test_get_naive_spreads(self):
        ntimes = 20
        nstars = 1000
        xyzuvw = np.zeros((nstars, ntimes, 6))
        dX = 5
        xyzuvw[:,:,:2] = np.random.randn(*xyzuvw[:,:,:2].shape) * dX

        naive_spreads = an.get_naive_spreads(xyzuvw)

        self.assertEqual(naive_spreads.shape[0], ntimes)
        self.assertTrue(np.isclose(naive_spreads, dX, 1e-1).all())

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(AnalyserTestCase)
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

sys.path.insert(0, '.')  # reinserting home directory into path

