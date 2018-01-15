#!/usr/bin/env python -W ignore
"""
quadplotter_test.py
----------------------------

Tests for `quadplotter` module.

To Do:
"""

import sys

sys.path.insert(0, '..')  # hacky way to get access to module

import chronostar.synthesiser as syn
import chronostar.traceback as tb
import chronostar.quadplotter as qp
import numpy as np
import pickle


def test_basic_plot():
    synth_file = 'quad_data.pkl'
    tb_file = 'tb_quad_data.pkl'

    mock_twa_pars = [
        -80, 80, 50, 10, -20, -5, 5, 5, 5, 2, 0.0, 0.0, 0.0, 7, 40
    ]

    naive_spreads = np.array([
        14.68241966, 12.70310363, 10.77880051, 8.94765731,
        7.28784439, 5.96099301, 5.22924972, 5.27102879,
        6.02073234, 7.29844017, 8.90637535
    ])
    #naive_spreads = None

    bayes_spreads = np.array([
        15.70188242, 13.75653809, 11.55497521, 9.70393454,
        8.00953719, 6.45559187, 5.47807609, 5.60783995,
        6.37361751, 7.8988234, 9.51225275
    ])
    #bayes_spreads = None

    time_probs = np.array([
        0.00000000e+000,   0.00000000e+000,   4.94065646e-324,
        1.15117048e-317,   1.21721962e-310,   4.07224337e-303,
        7.24237846e-298,   2.16716626e-301,   1.34741822e-305,
        3.23673453e-311,   2.03703266e-318
    ])
    time_probs = None

    #times = np.array([
    #    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.
    #])

    mock_twa_pars = [
        -80, 80, 50, 10, -20, -5, 5, 5, 5, 2, 0.0, 0.0, 0.0, 7, 40
    ]
    error = 1

    try:
        with open(synth_file, 'r') as fp:
            t = pickle.load(fp)
    except IOError:
        syn.synthesise_data(
            1, mock_twa_pars, error, savefile=synth_file
        )
        with open(synth_file, 'r') as fp:
            t = pickle.load(fp)

    try:
        with open(tb_file, 'r') as fp:
            pass
    except IOError:
        times = np.linspace(0, 10, 11)
        tb.traceback(t, times, savefile=tb_file)

    qp.plot_quadplots(
        tb_file, bayes_spreads=bayes_spreads, time_probs=time_probs,
        naive_spreads=naive_spreads, init_conditions=mock_twa_pars)


if __name__ == '__main__':
    test_basic_plot()

sys.path.insert(0, '.')  # reinserting home directory into path
