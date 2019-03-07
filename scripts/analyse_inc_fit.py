"""Use this script to conveniently load all aspects from an inc fit"""
from __future__ import print_function, division

import numpy as np

import os
import sys
sys.path.insert(0, '..')
import chronostar.retired2.datatool as dt


def loadFinalResults(fdir):
    fgroups = dt.loadGroups(fdir + 'final_groups.npy')
    fmembs = np.load(fdir + 'final_membership.npy')
    fmed_errs = np.load(fdir + 'final_med_errs.npy')
    return fgroups, fmembs, fmed_errs

assoc_name = sys.argv[1]
if os.path.isdir('/data/mash/tcrun/'):
    rdir = "/data/mash/tcrun/em_fit/{}/".format(assoc_name)
else:
    rdir = "../results/em_fit/{}/".format(assoc_name)

xyzuvw_file = "../data/{}_xyzuvw.fits".format(assoc_name)

star_pars = dt.loadXYZUVW(xyzuvw_file)

ncomps = 1
# assert true_memb.shape[0] == dt.loadXYZUVW(star_pars_file)['xyzuvw'].shape[0]
# print("true memb shape: {}".format(true_memb.shape))
# print("nstars: {}".format(dt.loadXYZUVW(star_pars_file)['xyzuvw'].shape[0]))
final_fits = []
final_membs = []
final_med_errs = []
while os.path.isdir(rdir + '{}/'.format(ncomps)):
    print("ncomps: {}".format(ncomps))
    if ncomps == 1:
        fgroups, fmembs, fmed_errs = loadFinalResults(rdir + '1/final/')
        final_fits.append(fgroups)
        final_membs.append(fmembs)
        final_med_errs.append(fmed_errs)
    else:
        comp_fgroups = []
        comp_fmembs = []
        comp_fmed_errs = []
        for i in range(ncomps-1):
            subrdir = rdir + '{}/{}/final/'.format(ncomps, chr(ord('A') + i))
            if os.path.isdir(subrdir):
                fgroups, fmembs, fmed_errs = loadFinalResults(subrdir)
                comp_fgroups.append(fgroups)
                comp_fmembs.append(fmembs)
                comp_fmed_errs.append(fmed_errs)
        final_fits.append(comp_fgroups)
        final_membs.append(comp_fmembs)
        final_med_errs.append(comp_fmed_errs)
    ncomps += 1

# load in the synthetic data
sdir = rdir + 'synth_data/'
if os.path.isdir(sdir):
    origins = dt.loadGroups(sdir + 'origins.npy')
    perf_xyzuvw = np.load(sdir + 'perf_xyzuvw.npy')
    true_z = dt.getZfromOrigins(origins, star_pars)
    nassoc_stars = np.sum([o.nstars for o in origins])

try:
    bg_ln_ols = np.load(rdir + 'bg_ln_ols.npy')
except IOError:
    print("couldn't find background ln overlaps...")

