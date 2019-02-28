from __future__ import division, print_function

import chronostar.synthdata

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


res_dir = '../results/measured/'

scenarios = [
    '5_5_2_25',
    '15_5_2_25',
    '30_5_2_25',
    '50_5_2_25',
]

precs = ['perf', 'half', 'gaia', 'doub', 'trip']
prec_val = {'perf':1e-5, 'half':0.5, 'gaia':1.0, 'doub':2.0, 'trip':3.0,
            'quad':4.0, 'quin':5.0}
astro_tables = []
fits_files = []
for scen in scenarios:
    for prec in precs:
        fits_file = res_dir + "xyzuvw_now_" + scen + "_" + prec + ".fits"

        fits_files.append(fits_file)
        try:
            with open(fits_file, 'r') as fp:
                pass
        except IOError:
            #xyzuvw_file = res_dir + scen + "_xyzuvw_init.npy"
            xyzuvw_file = res_dir + scen + "_perf_xyzuvw.npy"
            init_xyzuvw = np.load(xyzuvw_file)
            astr_table = chronostar.synthdata.measureXYZUVW(init_xyzuvw, prec_val[prec])
            cv.convertMeasurementsToCartesian(t=astr_table,
                                              savefile=fits_file)



file_stems = [scen + "_" + prec for scen in scenarios for prec in precs]

#fit_files = [res_dir + "xyzuvw_now_" + stem + ".fits" for stem in file_stems]
assert len(fits_files) == len(file_stems)

for i in range(len(fits_files)):
    xyzuvw_dict = gf.loadXYZUVW(fits_files[i])
    nstars = xyzuvw_dict['xyzuvw'].shape[0]
    origin_file = res_dir + file_stems[i][:-5] + '_origins.npy'
    group = np.load(origin_file).item()
    true_age = group.age
    ntimes = int(2 * true_age + 1)
    times = -np.linspace(0, 2 * true_age, ntimes)
    tb = torb.traceManyOrbitXYZUVW(xyzuvw_dict['xyzuvw'], times=times,
                                   single_age=False,
                                   savefile=res_dir + 'tb_{}.npy'.\
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
    plt.savefig(res_dir + file_stems[i]+"_sep_from_mean.pdf")
