from __future__ import division, print_function

import chronostar.synthdata

"""
Simple script generating some orbital traceback plots, represented
as separation from mean.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.traceorbit as torb
import integration_tests.traceback_plotter as tp
import chronostar.groupfitter as gf
import chronostar.measurer as ms
import chronostar.converter as cv

rdir = '../results/synth_fit/50_2_1_50/'
sdir = 'temp_data/'
xyzuvw_now_file = rdir + 'gaia/xyzuvw_now.fits'
origin_file = rdir + 'origins.npy'
perf_xyzuvw_file = rdir + 'perf_xyzuvw.npy'

origin = np.load(origin_file).item()
perf_xyzuvw_now = np.load(perf_xyzuvw_file)

maxtime = 2*origin.age
ntimes = maxtime + 1
times = np.linspace(0, np.int(-maxtime), ntimes)
#xyzuvw_dict = gf.loadXYZUVW(xyzuvw_now_file)

precs = ['perf', 'half', 'gaia', 'double']
prec_val = {'perf':1e-5, 'half':0.5, 'gaia':1.0, 'double':2.0}

for prec in precs:
    astro_table = chronostar.synthdata.measureXYZUVW(perf_xyzuvw_now, prec_val[prec])
    xyzuvw_dict = cv.convertMeasurementsToCartesian(astro_table)
    # for each star, sample its possible phase properties... 10 times?
    tp.plotSeparation(xyzuvw_dict['xyzuvw'], times, prec=prec)

    mc_xyzuvws = np.zeros((0,6))
    nsamples = 10
    for (mn, cov) in zip(xyzuvw_dict['xyzuvw'], xyzuvw_dict['xyzuvw_cov']):
        samples = np.random.multivariate_normal(mn, cov, size=nsamples)
        mc_xyzuvws = np.vstack((mc_xyzuvws, samples))

    tp.plotSeparation(mc_xyzuvws, times, prec=prec + '_emcee')
