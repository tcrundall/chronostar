#!/usr/bin/env python

import sys
sys.path.insert(0,'..')

import numpy as np
import pickle
import chronostar.tracingback as tb

restricted_data_file = \
    "../data/rave_active_stars_distance_better_than_5_percent_3k_stars.pkl"
complete_data_file = \
    "../data/rave_active_stars_distance_better_than_20_percent_16k_stars.pkl"
infiles = [restricted_data_file, complete_data_file]

times = np.array([0,1.0])
#times = np.array([0.0]) <-- in order to do this, need to add shortcut
# in interpolate_cov section, since it tries to interpolate form
# time 0 to time 1 anyways
for infile in infiles:
    savefile = "../data/tb_" + infile[8:]
    with open(infile, 'r') as fp:
        t = pickle.load(fp)
    assert type(t['Name'][0]) == np.string_
    tb.traceback(
        t, times,
        savefile=savefile,
    )

