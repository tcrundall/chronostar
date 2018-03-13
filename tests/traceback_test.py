#!/usr/bin/env python -W ignore
"""
traceback_test.py
----------------------------

Tests for `traceback` module.
"""

import os.path
import sys
import tempfile
import unittest

sys.path.insert(0, '..')  # hacky way to get access to module

import numpy as np
import chronostar.traceback as tb
import pdb
import pickle


class TracebackTestCase(unittest.TestCase):
    def setUp(self):

        self.tempdir = tempfile.mkdtemp()
        self.tb_file = os.path.join(self.tempdir, 'tb_data.pkl')

        mock_twa_pars = [
            -80, 80, 50, 10, -20, -5, 5, 5, 5, 2, 0.0, 0.0, 0.0, 7, 40
        ]

        NGROUPS = 1
#
#    def tearDown(self):
#        try:
#            os.remove(self.synth_file)
#        except OSError, AttributeError:
#            pass
#        try:
#            os.remove(self.tb_file)
#        except OSError, AttributeError:
#            pass
#        try:
#            os.rmdir(self.tempdir)
#        except AttributeError:
#            pass

    def test_get_naive_spreads(self):
        ntimes = 20
        nstars = 1000
        xyzuvw = np.zeros((nstars, ntimes, 6))
        dX = 5
        xyzuvw[:,:,:2] = np.random.randn(*xyzuvw[:,:,:2].shape) * dX

        naive_spreads = an.get_naive_spreads(xyzuvw)

        self.assertEqual(naive_spreads.shape[0], ntimes)
        self.assertTrue(np.isclose(naive_spreads, dX, 1e-1).all())

    def test_sky_coords(self):
        """
        Handy reference on how to utilise traceback functions...
        :return:
        """
        # beta pic coordinates: (radecpipmrv)
        bp_astr = [86.82, -51.067, 51.44, 4.65, 83.1, 20]
        age = 20

        times = np.array([0.0, age])
        bp_xyzuvws = tb.integrate_xyzuvw(bp_astr, times)
        bp_xyzuvw_now = bp_xyzuvws[0]
        bp_xyzuvw_then = bp_xyzuvws[1]

        assert(np.allclose(
            tb.xyzuvw_to_skycoord(bp_xyzuvw_now, 'schoenrich', True),
            bp_astr, rtol=1e-5
        ))

        bp_xyzuvw_now_same = tb.trace_forward(
            bp_xyzuvw_then, age, solarmotion=None
        )
        assert np.allclose(bp_xyzuvw_now, bp_xyzuvw_now_same, rtol=1e-5)

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TracebackTestCase)
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

sys.path.insert(0, '.')  # reinserting home directory into path

