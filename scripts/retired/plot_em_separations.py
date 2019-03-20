from __future__ import division, print_function
"""
Simple script generating some orbital traceback plots, represented
as separation from mean.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.traceorbit as torb
import integration_tests.traceback_plotter as tp
import chronostar.compfitter as gf
import chronostar.transform as tf

rdir = '../results/em_fit/field_blind/'
sdir = 'temp_data/'

origin_file = rdir + 'origins.npy'
perf_xyzuvw_file = rdir + 'perf_xyzuvw.npy'
memberships_file = rdir + 'memberships.npy'
final_groups_file = rdir + 'final_groups.npy'
xyzuvw_file = rdir + 'xyzuvw_now.fits'

origins = np.load(origin_file)
perf_xyzuvw_now = np.load(perf_xyzuvw_file)
z_final = np.load(memberships_file)
final_groups = np.load(final_groups_file)
xyzuvw_dict = gf.loadXYZUVW(xyzuvw_file)


ass_cov_now = tf.transform_covmatrix(final_groups[0].generateCovMatrix(),
                                     torb.traceOribtXYZUVW,
                                     final_groups[0].mean,
                                     args=(final_groups[0].age,))
ass_mn_now = torb.traceOribtXYZUVW(final_groups[0].mean,
                                   final_groups[0].age)
simple_cov_now = np.copy(ass_cov_now)
simple_cov_now[3:6,:3] = 0
simple_cov_now[:3,3:6] = 0

members_mask = np.arange(50)
field_mask = np.arange(50,1050)
nstars = z_final.shape[0]
precs = ['perf', 'half', 'gaia', 'double']
prec_val = {'perf':1e-5, 'half':0.5, 'gaia':1.0, 'double':2.0}

prec = 'gaia'

# astro_table = ms.measureXYZUVW(perf_xyzuvw_now, prec_val[prec])
# xyzuvw_dict = cv.convertMeasurementsToCartesian(astro_table)

simple_lnols = gf.get_lnoverlaps(simple_cov_now, ass_mn_now,
                                 xyzuvw_dict['xyzuvw_cov'],
                                 xyzuvw_dict['xyzuvw'], nstars)

# only lose 5% of true members
candidate_member_mask = np.where(simple_lnols >
                                np.percentile(simple_lnols[:50], 5))

maxtime = 2*origins[0].age
assert maxtime > 1
ntimes = maxtime + 1
times = np.linspace(0, np.int(-maxtime), ntimes)
#xyzuvw_dict = gf.loadXYZUVW(xyzuvw_now_file)
#
# precs = ['perf', 'half', 'gaia', 'double']
# prec_val = {'perf':1e-5, 'half':0.5, 'gaia':1.0, 'double':2.0}
#
# for prec in precs:
#astro_table = ms.measureXYZUVW(perf_xyzuvw_now, prec_val[prec])
#xyzuvw_dict = cv.convertMeasurementsToCartesian(astro_table)
# for each star, sample its possible phase properties... 10 times?
tp.plotSeparation(xyzuvw_dict['xyzuvw'][candidate_member_mask],
                  times, prec=prec + '_em')

mc_xyzuvws = np.zeros((0,6))
nsamples = 10
for (mn, cov) in zip(xyzuvw_dict['xyzuvw'], xyzuvw_dict['xyzuvw_cov']):
    samples = np.random.multivariate_normal(mn, cov, size=nsamples)
    mc_xyzuvws = np.vstack((mc_xyzuvws, samples))

tp.plotSeparation(mc_xyzuvws, times, prec=prec + '_emcee_em')
