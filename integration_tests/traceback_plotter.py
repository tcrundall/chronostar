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
import chronostar.measurer as ms
import chronostar.converter as cv


def get_euc_dist(pos1, pos2):
    """pos1 and pos2 are two (x,y,z) tuples"""
    diff = pos1 - pos2
    return r_from_xyz(diff)


def r_from_xyz(xyzs):
    xyzs = np.copy(xyzs)
    if len(xyzs.shape) == 1:
        xyzs = np.array([xyzs])
    return np.sqrt(np.sum(xyzs ** 2, axis=1))


rdir = 'temp_data/'

precs = ['perf', 'half', 'gaia', 'double', 'triple', 'quad', 'quint']
prec_val = {'perf':1e-5, 'half':0.5, 'gaia':1.0, 'double':2.0, 'triple':3.0,
            'quad':4.0, 'quint':5.0}
astro_table_files = [rdir + prec+'_astro_table.txt' for prec in precs]
fits_files = [rdir + prec+'_xyzuvw_now.fits' for prec in precs]

origin_file = rdir + 'origins.npy'

for i, prec in enumerate(precs):
    xyzuvw_dict = gf.loadXYZUVW(fits_files[i])
    nstars = xyzuvw_dict['xyzuvw'].shape[0]
    group = np.load(origin_file).item()
    true_age = group.age
    ntimes = int(2 * true_age + 1)
    times = -np.linspace(0, 2 * true_age, ntimes)
    tb = torb.traceManyOrbitXYZUVW(xyzuvw_dict['xyzuvw'], times=times,
                                   single_age=False, savefile='tb_{}.npy'. \
                                   format(int(true_age)))

    # for each timestep, get mean of association and distance from mean
    dists = np.zeros((nstars, ntimes))
    stds = np.zeros(ntimes)
    for tix in range(ntimes):
        dists[:, tix] = get_euc_dist(tb[:, tix, :3],
                                     tb[:, tix, :3].mean(axis=0))

    plt.clf()
    plt.plot(-times, dists.T, '#888888')
    plt.plot(-times, np.median(dists.T, axis=1), 'r')
    plt.savefig(prec+"_sep_from_mean.png")
