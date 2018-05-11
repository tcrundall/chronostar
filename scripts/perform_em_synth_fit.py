#! /usr/bin/env python

try:
    import matplotlib as mpl
    mpl.use('Agg')
except ImportError:
    pass

from distutils.dir_util import mkpath
import logging
import numpy as np
import os
import platform
import pickle
import sys
sys.path.insert(0, '..')

import chronostar.synthesiser as syn
import chronostar.traceorbit as torb
import chronostar.converter as cv
import chronostar.measurer as ms
import chronostar.groupfitter as gf

dir_name = sys.argv[1]

try:
    rdir = "/data/mash/tcrun/em_fit/" + dir_name
    mkpath(rdir)
except IOError:
    print("I'm guessing you're not Tim Crundall... nor on an RSAA server")
    rdir = "../results/em_fit/" + dir_name
    mkpath(rdir)
#os.chdir(res_dir)

logging.basicConfig(
    level=logging.DEBUG, filemode='w',
    filename='em.log',
)

# Setting up standard filenames
xyzuvw_perf_file     = "perf_xyzuvw.npy"
#group_savefile       = 'origins.npy'
xyzuvw_init_savefile = 'xyzuvw_init.npy'
astro_savefile       = 'astro_table.txt'
xyzuvw_conv_savefile = 'xyzuvw_now.fits'

mean_now = np.array([50., -100., -0., -10., -20., -5.])
extra_pars = np.array([
    #dX, dV, age, nstars
    [10., 3., 10., 20.],
    [10., 5.,  7., 100.],
])

#origins = np.array([
#   #  X    Y    Z    U    V    W   dX  dY    dZ  dVCxyCxzCyz age nstars
#   [25., 0., 11., -5., 0., -2., 10., 10., 10., 5., 0., 0., 0., 3., 50.],
#   [-21., -60., 4., 3., 10., -1., 7., 7., 7., 3., 0., 0., 0., 7., 30.],
##  [-10., 20., 0., 1., -4., 15., 10., 10., 10., 2., 0., 0., 0., 10., 40.],
##  [-80., 80., -80., 5., -5., 5., 20., 20., 20., 5., 0., 0., 0., 13., 80.],
#])
ERROR = 1.0
ngroups = extra_pars.shape[0]

logging.info("Mean (now):\n{}".format(mean_now))
logging.info("Extra pars:\n{}".format(extra_pars))
#np.save("origins.npy", origins)
groups = []

for i in range(ngroups):
    group_pars = np.hstack(mean_now, extra_pars[i])
    perf_xyzuvws, group = syn.generate_current_pos(group_pars, sphere=True,
                                                   )
    groups.append(group)

np.save("perf_xyzuvw.npy", perf_xyzuvws)
sky_coord_now = syn.measure_stars(perf_xyzuvws)

synth_table = syn.generate_table_with_error(
    sky_coord_now, ERROR
)

pickle.dump(synth_table, open(astro_savefile, 'w'))
tb.traceback(synth_table, np.array([0, 1]), savefile=TB_FILE)
star_pars = tfgf.read_stars(TB_FILE)

tfem.fit_multi_groups(star_pars, ngroups)
