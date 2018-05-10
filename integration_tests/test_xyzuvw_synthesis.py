from __future__ import division, print_function
"""
Confirm reasonable deviation in XYZUVW due to synthetic measurement error
"""

import logging
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

#import astropy.io.fits as pyfits
from astropy.table import Table

sys.path.insert(0, '..')

import chronostar.synthesiser as syn
import chronostar.measurer as ms
import chronostar.converter as cv
import chronostar.traceorbit as torb
import chronostar.errorellipse as ee
import chronostar.transform as tf

logging.basicConfig(level=logging.INFO, filemode='a',
                    filename='logs/test_xyzuvw_synthesis.log')

rdir = "temp_data/"
group_savefile = 'origins.npy'
xyzuvw_init_savefile = 'xyzuvw_init.npy'
xyzuvw_perf_file = 'perf_xyzuvw.npy'

precs = ['perf', 'half', 'gaia', 'double', 'triple', 'quad', 'quint']
prec_val = {'perf':1e-5, 'half':0.5, 'gaia':1.0, 'double':2.0,
            'triple':3.0, 'quad':4.0, 'quint':5.0}

age = 10
dX = 10
dV = 2
nstars = 30

mean_now = np.array([0., -300., 0., -10., -20., -5.])
mean_then = torb.traceOrbitXYZUVW(mean_now, -age)
group_pars = np.hstack((mean_then, dX, dV, age, nstars))

xyzuvw_init, origin = \
    syn.synthesise_xyzuvw(group_pars, sphere=True,
                          xyzuvw_savefile=rdir + xyzuvw_init_savefile,
                          return_group=True,
                          group_savefile=rdir + group_savefile)

# Sanity check, test stars traceback to approximately origin
print("Mean of init stars close to initialisation? {}".format(
    np.allclose(origin.mean, np.mean(xyzuvw_init, axis=0), atol=5.)
))
print("Mean of init stars matches initialisation? {}".format(
    np.allclose(origin.mean, np.mean(xyzuvw_init, axis=0), atol=2.)
    ))
print("dx of init stars close to initialisation? {}".format(
    np.allclose(origin.dx, np.std(xyzuvw_init[:,:3], axis=0), atol=5.)
))
print("dx of init stars matches initialisation? {}".format(
    np.allclose(origin.dx, np.std(xyzuvw_init[:,:3], axis=0), atol=2.)
))
print("dv of init stars close to initialisation? {}".format(
    np.allclose(origin.dv, np.std(xyzuvw_init[:,3:], axis=0), atol=5.)
))
print("dv of init stars matches initialisation? {}".format(
    np.allclose(origin.dv, np.std(xyzuvw_init[:,3:], axis=0), atol=2.)
))

xyzuvw_now_perf = torb.traceManyOrbitXYZUVW(xyzuvw_init, origin.age,
                                            single_age=True,
                                            savefile=rdir+xyzuvw_perf_file)

print("Mean of stars now close to origin seed: {}".format(
    np.allclose(mean_now, np.mean(xyzuvw_now_perf, axis=0), atol=10.)
))
print("Mean of stars now matches origin seed: {}".format(
    np.allclose(mean_now, np.mean(xyzuvw_now_perf, axis=0), atol=5.)
))
perf_astro_filename = rdir + 'perf_astro_table.txt'
perf_xyzuvw_conv_savefile = rdir + 'perf_xyzuvw_now.fits'
perf_astro_table = ms.measureXYZUVW(xyzuvw_now_perf, prec_val['perf'],
                                    savefile=perf_astro_filename)
perf_star_pars = cv.convertMeasurementsToCartesian(
    perf_astro_table, savefile=perf_xyzuvw_conv_savefile
)

for prec in precs:
    if prec != 'perf':
        plt.clf()
        plt.plot(perf_star_pars['xyzuvw'][:,0], perf_star_pars['xyzuvw'][:,1],
                 '.', label='perf now' )
        astro_savefile = rdir + prec + "_astro_table.txt"
        xyzuvw_conv_savefile = rdir + prec + "_xyzuvw_now.fits"
        astro_table = ms.measureXYZUVW(xyzuvw_now_perf, prec_val[prec],
                                       savefile=astro_savefile)

        # confirm deviation from perfect astrometry is consistent
        rv_diff = astro_table['rv'] - perf_astro_table['rv']
        print(np.isclose(prec_val[prec]*ms.GERROR['e_RV'],
                        rv_diff.std(), rtol=0.3))
        plx_diff = astro_table['plx'] - perf_astro_table['plx']
        print(np.isclose(prec_val[prec]*ms.GERROR['e_Plx'],
                        plx_diff.std(), rtol=0.3))
        pmde_diff = astro_table['pmde'] - perf_astro_table['pmde']
        print(np.isclose(prec_val[prec]*ms.GERROR['e_pm'],
                        pmde_diff.std(), rtol=0.3))
        pmra_diff = astro_table['pmra'] - perf_astro_table['pmra']
        print(np.isclose(prec_val[prec]*ms.GERROR['e_pm'],
                        pmra_diff.std(), rtol=0.3))
        print("Astrometry error ranges are within 33% (atleast) of"
              " provided errors")

        star_pars = cv.convertMeasurementsToCartesian(
            astro_table, savefile=xyzuvw_conv_savefile
        )

        # check current distribution is what we tried to start with
        print(np.allclose(mean_now, star_pars['xyzuvw'].mean(axis=0),
                           atol=10))
#        plt.plot(star_pars['xyzuvw'][:,0],
#                 star_pars['xyzuvw'][:,1],
#                 '.', label=prec
#                 )
        for (mn, cov) in zip(star_pars['xyzuvw'],star_pars['xyzuvw_cov']):
            # plot now
            ee.plotCovEllipse(cov[:2,:2],mn[:2],
                              nstd=2,alpha=0.1,color='b')

            # plot then
            mn_then = torb.traceOrbitXYZUVW(mn, -age)
            cov_then = tf.transform_cov(cov, torb.traceOrbitXYZUVW, mn,
                                        args=(-age,))

            ee.plotCovEllipse(cov_then[:2,:2],mn_then[:2],
                              nstd=2,alpha=0.03,color='r')

        times = np.linspace(0,-age,int(age)+1)
        tracedback_xyzuvws = torb.traceManyOrbitXYZUVW(star_pars['xyzuvw'],
                                                       times,single_age=False)
        plt.plot(tracedback_xyzuvws[:,:,0].T, tracedback_xyzuvws[:,:,1].T,
                 'b', alpha=0.05)

        plt.plot(xyzuvw_init[:,0], xyzuvw_init[:,1], 'm.', label='perf then')


        plt.legend(loc='best')
        plt.savefig("all_stars_{}.pdf".format(prec))
