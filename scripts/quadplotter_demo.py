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
import chronostar.tracingback as tb
import chronostar.quadplotter as qp
import numpy as np
import pickle


def build_basic_plot():
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
    naive_spreads = None

    bayes_spreads = np.array([
        15.70188242, 13.75653809, 11.55497521, 9.70393454,
        8.00953719, 6.45559187, 5.47807609, 5.60783995,
        6.37361751, 7.8988234, 9.51225275
    ])
    bayes_spreads = None

    time_probs = np.array([
        6.06109261e-37, 3.21525005e-32, 1.14692564e-26,
        4.42139528e-20, 8.48183836e-13, 1.27153372e-05,
        1.00000000e+00, 1.00832735e-01, 1.18302909e-07,
        3.58372676e-13, 5.03060168e-20
    ])
    time_probs = None

    times = np.array([
        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.
    ])
    times = np.linspace(0,10,21)

    # CANNOT DO SOMETHING LIKE THIS, TOO MANY FUNCTIONS EXPECT TIMES TO BE
    # LINEARLY SPACED TODO: FIX LINEAR EXPECTATION OF TIMES
#    times = np.append(
#        np.append(
#            np.linspace(0, 6, 6, endpoint=False),
#            np.linspace(6, 8, 20, endpoint=False)
#        ),
#        np.linspace(8, 10, 3)
#    )

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
        tb.traceback(t, times, savefile=tb_file)

    qp.plot_quadplots(
        tb_file, bayes_spreads=bayes_spreads, time_probs=time_probs,
        naive_spreads=naive_spreads, init_conditions=mock_twa_pars,
        plot_it=True
    )


if __name__ == '__main__':
    build_basic_plot()
