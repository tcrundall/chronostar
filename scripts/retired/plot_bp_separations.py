from __future__ import division, print_function
"""
Simple script generating some orbital traceback plots, represented
as separation from mean.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

import integration_tests.traceback_plotter as tp
import chronostar.compfitter as gf

rdir = '../results/em_fit/gaia_dr2_bp/'
sdir = rdir
xyzuvw_now_file = '../data/gaia_dr2_bp_xyzuvw.fits'
final_z_file = rdir + 'final/final_membership.npy'

maxtime = 30
ntimes = maxtime + 1
times = np.linspace(0, np.int(-maxtime), ntimes)
final_z = np.load(final_z_file)

xyzuvw_dict = gf.loadXYZUVW(xyzuvw_now_file)
tp.plotSeparation(xyzuvw_dict['xyzuvw'][np.where(final_z[:,0])], times, prec='real-data')

mc_xyzuvws = np.zeros((0,6))
nsamples = 10
for (mn, cov) in zip(xyzuvw_dict['xyzuvw'][np.where(final_z[:,0])], xyzuvw_dict['xyzuvw_cov']):
    samples = np.random.multivariate_normal(mn, cov, size=nsamples)
    mc_xyzuvws = np.vstack((mc_xyzuvws, samples))

tp.plotSeparation(mc_xyzuvws, times, prec='real-data'+ '_emcee')
