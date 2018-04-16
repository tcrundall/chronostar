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
import pickle
import sys
sys.path.insert(0, '..')

import chronostar.synthesiser as syn
import chronostar.tracingback as tb
import chronostar.tfgroupfitter as tfgf
import chronostar.tfexpectmax as tfem

res_dir = sys.argv[1]
mkpath(res_dir)
os.chdir(res_dir)

logging.basicConfig(
    level=logging.DEBUG, filemode='w',
    filename='em.log',
)

origins = np.array([
   #  X    Y    Z    U    V    W   dX  dY    dZ  dVCxyCxzCyz age nstars
   [25., 0., 11., -5., 0., -2., 10., 10., 10., 5., 0., 0., 0., 3., 50.],
   [-21., -60., 4., 3., 10., -1., 7., 7., 7., 3., 0., 0., 0., 7., 30.],
#       [-10., 20., 0., 1., -4., 15., 10., 10., 10., 2., 0., 0., 0., 10., 40.],
#       [-80., 80., -80., 5., -5., 5., 20., 20., 20., 5., 0., 0., 0., 13., 80.],

])
ERROR = 1.0

ngroups = origins.shape[0]
TB_FILE = "perf_tb_file.pkl"
astr_file = "perf_astr_data.pkl"

logging.info("Origin:\n{}".format(origins))
np.save("origins.npy", origins)
perf_xyzuvws, _ = syn.generate_current_pos(ngroups, origins)

np.save("perf_xyzuvw.npy", perf_xyzuvws)
sky_coord_now = syn.measure_stars(perf_xyzuvws)

synth_table = syn.generate_table_with_error(
    sky_coord_now, ERROR
)

pickle.dump(synth_table, open(astr_file, 'w'))
tb.traceback(synth_table, np.array([0, 1]), savefile=TB_FILE)
star_pars = tfgf.read_stars(TB_FILE)

tfem.fit_multi_groups(star_pars, ngroups)
