from __future__ import division, print_function
"""
Reproduces the traceback plots seen all through the literature
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.groupfitter as gf
import chronostar.traceorbit as torb

def get_euc_dist(pos1, pos2):
    """pos1 and pos2 are two (x,y,z) tuples"""
    diff = pos1 - pos2
    return r_from_xyz(diff)

def r_from_xyz(xyzs):
    xyzs = np.copy(xyzs)
    if len(xyzs.shape) == 1:
        xyzs = np.array([xyzs])
    return np.sqrt(np.sum(xyzs**2, axis=1))

res_dir = '../results/measured/'

file_stems = [
    '15_5_2_25_gaia',
    '15_5_2_25_perf',
    '30_5_2_25_gaia',
    '30_5_2_25_perf',
    '5_5_2_25_gaia',
    '5_5_2_25_perf',
]
fit_files = [res_dir + "xyzuvw_now_" + stem + ".fits" for stem in file_stems]

for i, fit_file in enumerate([fit_files[3]]):
    xyzuvw_dict = gf.loadXYZUVW(fit_file)
    origin_file = res_dir + file_stems[i][:-5] + '_origins.npy'
    group = np.load(origin_file).item()
    true_age = group.age
    times = -np.linspace(0,2*true_age,2*true_age + 1)
    tb = torb.traceManyOrbitXYZUVW(xyzuvw_dict['xyzuvw'], times=times,
                                   single_age=False, savefile='tb_{}.npy'.\
                                   format(int(true_age)))

    # for each timestep, get mean of association and distance from mean
    dists_now = get_euc_dist(tb[:,0,:3], tb[:,0,:3].mean(axis=0))
    dists_origin = get_euc_dist(tb[:,15,:3], tb[:,15,:3].mean(axis=0))

