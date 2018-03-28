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
from chronostar import utils
import chronostar.transform as tf
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

    def test_traceforward_group(self):
        """
        Compares the traceforward of a group with a covariance matrix
        fitted to stars drawn from the initial sample.
        """
        nstars = 100
        age = 20.

        dummy_groups = [
            # X,Y,Z,U,V,W,dX,dY,dZ, dV,Cxy,Cxz,Cyz,age,
            [0, 0, 0, 0, 0, 0, 10, 10, 10, 2, 0., 0., 0., age],
            # isotropic expansion
            [0, 0, 0, 0, 0, 0, 10, 1, 1, .1, 0., 0., 0., 2 * age],
            # should rotate anticlock
            [-20, -20, 300, 0, 0, 0, 10, 10, 10, 2, 0., 0., 0., age],
            # isotropic expansion
        ]

        for cnt, dummy_group_pars_ex in enumerate(dummy_groups):
            mean = dummy_group_pars_ex[0:6]
            cov = utils.generate_cov(
                utils.internalise_pars(dummy_group_pars_ex)
            )
            stars = np.random.multivariate_normal(mean, cov, nstars)

            new_stars = np.zeros(stars.shape)
            for i, star in enumerate(stars):
                new_stars[i] = tb.trace_forward(star, age)

            # calculate the new mean and cov
            new_mean = tb.trace_forward(mean, age)
            new_cov = tf.transform_cov(
                cov, tb.trace_forward, mean, dim=6, args=(age,)
            )
            new_eigvals = np.linalg.eigvalsh(new_cov)

            estimated_cov = np.cov(new_stars.T)
            estimated_eigvals = np.linalg.eigvalsh(estimated_cov)

            assert np.allclose(new_eigvals, estimated_eigvals, rtol=.5)

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TracebackTestCase)
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

