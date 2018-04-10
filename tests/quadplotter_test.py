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

import chronostar.synthesiser as syn
import chronostar.tracingback as tb
import chronostar.quadplotter as qp
import numpy as np
import pickle


class QuadplotterTestCase(unittest.TestCase):
    def setUp(self):
        self.synth_file = 'quad_data.pkl'
        self.tb_file = 'tb_quad_data.pkl'

        mock_twa_pars = [
            -80, 80, 50, 10, -20, -5, 5, 5, 5, 2, 0.0, 0.0, 0.0, 7, 40
        ]

        self.naive_spreads = np.array([
            14.68241966, 12.70310363, 10.77880051, 8.94765731,
            7.28784439, 5.96099301, 5.22924972, 5.27102879,
            6.02073234, 7.29844017, 8.90637535
        ])

        self.bayes_spreads = np.array([
            15.70188242, 13.75653809, 11.55497521, 9.70393454,
            8.00953719, 6.45559187, 5.47807609, 5.60783995,
            6.37361751, 7.8988234, 9.51225275
        ])

        self.init_conditions = [
            -80, 80, 50, 10, -20, -5, 5, 5, 5, 2, 0.0, 0.0, 0.0, 7, 40
        ]

        self.times = np.array([
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.
        ])

    def test_basic_plot(self):
        mock_twa_pars = [
            -80, 80, 50, 10, -20, -5, 5, 5, 5, 2, 0.0, 0.0, 0.0, 7, 40
        ]
        error = 0.01

        try:
            with open(self.synth_file, 'r') as fp:
                t = pickle.load(fp)
        except IOError:
            syn.synthesise_data(
                1, mock_twa_pars, error, savefile=self.synth_file
            )
            with open(self.synth_file, 'r') as fp:
                t = pickle.load(fp)

        try:
            with open(self.tb_file, 'r') as fp:
                pass
        except IOError:
            times = np.linspace(0,10,11)
            tb.traceback(t, times, savefile=self.tb_file)

        qp.plot_quadplots(
            self.tb_file, bayes_spreads=self.bayes_spreads,
            naive_spreads=self.naive_spreads, init_conditions=mock_twa_pars)

        self.assertTrue(True)


def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(QuadplotterTestCase)
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

sys.path.insert(0, '.')  # reinserting home directory into path

