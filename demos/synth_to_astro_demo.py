#! /usr/bin/env python

import logging
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.synthesiser as syn
import chronostar.traceorbit as to
import chronostar.measurer as ms
import chronostar.converter as cv
import chronostar.coordinate as cc
import chronostar.errorellipse as ee
import chronostar.transform as tf


plot_it=True

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

save_dir = 'temp_data/'
group_savefile = save_dir + 'origins.npy'
xyzuvw_init_savefile = save_dir + 'xyzuvw_init.npy'
astro_savefile = save_dir + 'astro_table.txt'


group_pars = [0., 0., 0., 0., 0., 0., 1., 1., 0.5, 20]
xyzuvw_init, group = syn.synthesiseXYZUVW(
    group_pars, sphere=True, xyzuvw_savefile=xyzuvw_init_savefile,
    group_savefile=group_savefile, return_group=True
)
logging.info("Age is: {} Myr".format(group.age))
xyzuvw_now_true = to.traceManyOrbitXYZUVW(xyzuvw_init, np.array([0., group.age]))[:,1]
#assert np.allclose(np.mean(xyzuvw_now, axis=0), group.mean, rtol=1e-1)
logging.info("Mean of initial stars: {}".format(np.mean(xyzuvw_init, axis=0)))
logging.info("Mean of final stars: {}".format(np.mean(xyzuvw_now_true, axis=0)))

star_table = ms.measureXYZUVW(xyzuvw_now_true, 20.0, astro_savefile)
astr_arr, err_arr = ms.convertTableToArray(star_table)
nstars = len(star_table)

astr_covs = cv.convertAstroErrsToCovs(err_arr)

xyzuvw_now = cc.convertManyAstrometryToLSRXYZUVW(astr_arr, mas=True)
logging.info("Mean of retrieved stars: {}".format(np.mean(xyzuvw_now, axis=0)))

if plot_it:
    plt.clf()
    plt.plot(xyzuvw_now[:,1], xyzuvw_now[:,2], '.')
    ee.plotCovEllipse(group.generateEllipticalCovMatrix()[1:3,1:3],
                      group.mean[1:3], with_line=True)

xyzuvw_covs = np.zeros((nstars,6,6))
for ix in range(nstars):
    xyzuvw_covs[ix] = tf.transformCovMat(
        astr_covs[ix], cc.convertAstrometryToLSRXYZUVW, astr_arr[ix], dim=6
    )

if plot_it:
    for ix in range(nstars):
        ee.plotCovEllipse(xyzuvw_covs[ix,1:3,1:3], xyzuvw_now[ix,1:3], color='blue')
    plt.savefig("temp_plots/demo_run.png")
