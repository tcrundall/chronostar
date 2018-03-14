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
        pass

    def tearDown(self):
        pass

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

    def test_traceforward(self):
        bp_astr = [86.82, -51.067, 51.44, 4.65, 83.1, 20]
        age = 2
        times = np.array([0., age])

        bp_xyzuvws = tb.integrate_xyzuvw(bp_astr, times)
        bp_xyzuvw_now = bp_xyzuvws[0]
        bp_xyzuvw_then = bp_xyzuvws[1]

        bp_xyzuvw_now_same = tb.trace_forward(bp_xyzuvw_then, age, solarmotion=None)
        assert np.allclose(bp_xyzuvw_now, bp_xyzuvw_now_same)

        # tracing forward via a mid point
        half_age = 0.5 * age
        bp_xyzuvw_mid = tb.trace_forward(bp_xyzuvw_then, half_age, solarmotion=None)
        bp_xyzuvw_now_from_mid = tb.trace_forward(bp_xyzuvw_mid, half_age, solarmotion=None)
        assert np.allclose(bp_xyzuvw_now, bp_xyzuvw_now_from_mid, atol = 1e-3)

        # tracing forward via arbitrarily small steps
        ntimes = 101
        tstep = float(age) / (ntimes - 1)

        bp_xyzuvw_many = np.zeros((ntimes, 6))
        bp_xyzuvw_many[0] = bp_xyzuvw_then

        for i in range(1, ntimes):
            bp_xyzuvw_many[i] = tb.trace_forward(bp_xyzuvw_many[i - 1], tstep,
                                                 solarmotion=None)
        assert np.allclose(bp_xyzuvw_now, bp_xyzuvw_many[-1], atol=5e-3)

        # tracing forward for a larger age, same step size
        larger_age = 10 * age
        larger_ntimes = 10 * (ntimes - 1) + 1
        larger_tstep = float(larger_age) / (larger_ntimes - 1)

        bp_xyzuvw_many_larger = np.zeros((larger_ntimes, 6))
        bp_xyzuvw_then_larger = \
            tb.integrate_xyzuvw(bp_astr, np.array([0., larger_age]))[1]
        bp_xyzuvw_many_larger[0] = bp_xyzuvw_then_larger

        for i in range(1, larger_ntimes):
            bp_xyzuvw_many_larger[i] = tb.trace_forward(
                bp_xyzuvw_many_larger[i - 1], larger_tstep, solarmotion=None)

        # !!! for a larger age, but same step size, cannot retrieve original XYZUVW values
        self.assertFalse(np.allclose(bp_xyzuvw_now, bp_xyzuvw_many_larger[-1], atol=1.))


def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TracebackTestCase)
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

