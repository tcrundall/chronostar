#!/usr/bin/env python -W ignore
"""
quadplotter_test.py
----------------------------

Tests for `quadplotter` module.

To Do:
"""

import os.path
import sys
import tempfile
import unittest

sys.path.insert(0, '..')  # hacky way to get access to module

import numpy as np
import pickle


class QuadplotterTestCase(unittest.TestCase):
    def setUp(self):

        self.tempdir = tempfile.mkdtemp()
        self.synth_file = os.path.join(self.tempdir, 'synth_data.pkl')
        self.tb_file = os.path.join(self.tempdir, 'tb_data.pkl')

        mock_twa_pars = [
            -80, 80, 50, 10, -20, -5, 5, 5, 5, 2, 0.0, 0.0, 0.0, 7, 40
        ]

        NGROUPS = 1
        xyzuvw_now, nstars = syn.generate_current_pos(NGROUPS, mock_twa_pars)
        #sky_coord_now = syn.measure_stars(xyzuvw_now)

        self.many_group_pars = np.array([
            # X, Y, Z, U,  V,  W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars
            [0, 0, 0, 0, 0, 0, 10, 10, 10, 5, .5, .2, .3, 20, 100],
            [20, 20, 20, 5, 5, 5, 10, 10, 10, 5, -.3, -.6, .2, 20, 100],
            [50, 50, 50, 0, 0, -10, 10, 10, 10, 5, -.8, .3, .2, 40, 100],
        ])

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


def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(QuadplotterTestCase)
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

sys.path.insert(0, '.')  # reinserting home directory into path

