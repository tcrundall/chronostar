#!/usr/bin/env python

import sys
sys.path.insert(0,'..')

import numpy as np
import pickle
import chronostar.traceback as tb


infile = "../data/rave_active_star_candidates_with_TGAS_kinematics.pkl"
times = np.array([0,1.0])
#times = np.array([0.0]) <-- in order to do this, need to add shortcut
# in interpolate_cov section, since it tries to interpolate form
# time 0 to time 1 anyways
with open(infile, 'r') as fp:
    t = pickle.load(fp)
tb.traceback(
    t, times,
    savefile="../data/tb_rave_active_star_candidates_with_TGAS_kinematics.pkl"
)

