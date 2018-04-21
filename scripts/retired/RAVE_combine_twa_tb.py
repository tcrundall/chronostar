#! /usr/bin/env python
import sys
sys.path.insert(0, '..')

import chronostar.retired.tracingback as tb
import numpy as np
import pickle
from astropy.table import Table

## Make an (astropy?) table, and store as .pkl file
#xyzuvw = tb.traceback(t,times,savefile=data_dir+'TWA_traceback_15Myr.pkl')

twa_tbfile = "../data/TWA_traceback_15Myr.pkl"
RAVE_tbfile = "../data/tb_rave_active_star_candidates_with_TGAS_kinematics.pkl"
with open(twa_tbfile, 'r') as fp:
    twa_table, _, twa_xyzuvw, twa_xyzuvw_cov = pickle.load(fp)

n_twastars = twa_xyzuvw.shape[0]

with open(RAVE_tbfile, 'r') as fp:
    RAVE_table, times, RAVE_xyzuvw, RAVE_xyzuvw_cov = pickle.load(fp)

t = Table()
for col in twa_table.keys():
    t[col] = np.append(RAVE_table[col], twa_table[col])

times = np.array([0.0,1.0])
savefile = "../data/tb_RAVE_twa_combined.pkl"
tb.traceback(t, times, savefile=savefile)

