#! /usr/bin/env python

import logging
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.synthesiser as syn
import chronostar.traceorbit as to
import chronostar.measurer as ms

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

save_dir = 'temp_data/'
group_savefile = save_dir + 'origins.npy'
xyzuvw_init_savefile = save_dir + 'xyzuvw_init.npy'
astro_savefile = save_dir + 'astro_table.txt'


group_pars = [0., 0., 0., 0., 0., 0., 1., 5., 10, 100]

xyzuvw_init, group = syn.synthesise_xyzuvw(
    group_pars, sphere=True, xyzuvw_savefile=xyzuvw_init_savefile,
    group_savefile=group_savefile, return_group=True
)


xyzuvw_now = to.traceManyOrbitXYZUVW(xyzuvw_init, np.array([0., group.age]))[:,1]

star_table = ms.measureXYZUVW(xyzuvw_now, 1.0, astro_savefile)

