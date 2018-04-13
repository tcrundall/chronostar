#!/usr/bin/env python -W ignore
"""
synthesise_demo
----------------------------
Synthesise a demo group, and generate a number (3) of traceback files
for varying degrees of measurment precision.
"""

import sys
import subprocess

sys.path.insert(0,'..') #hacky way to get access to module

import chronostar.synthesiser as syn
import chronostar.tracingback as tb
import chronostar.error_ellipse as ee
import logging
import numpy as np
import pdb
import pickle

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

generate_files = True
fit_bayes = True

#data_file_1 = "../results/synth_data_1.pkl"
#data_file_100 = "../results/synth_data_100.pkl"
#data_file_200 = "../results/synth_data_200.pkl"

#data_files = [data_file_1, data_file_100, data_file_200]

error_percs =['10', '100', '250']
data_files = {}
tb_files  = {}
for error_perc in error_percs:
    data_files[error_perc] = "../results/synth_data_{}.pkl".format(error_perc)
    tb_files[error_perc]= "../results/tb_synth_data_{}.pkl".format(error_perc)

#group_pars = [-100, 100, 0, 
# 1km/s ~~ 1pc/Myr
#                   X,      Y,      Z,  U,      V,      W
#mock_twa_pars = [-12.49, -42.28, 21.55, 9.87, -18.06, -4.52]
# * 7 Myr
mock_twa_pars = [-80,80,50,10,-20,-5,5,5,5,2,0.0,0.0,0.0,7,40]

NGROUPS = 1
xyzuvw_now, nstars = syn.generate_current_pos(NGROUPS, mock_twa_pars)
sky_coord_now = syn.measure_stars(xyzuvw_now)

synth_tables = {}
times = np.linspace(0,10,31)
# with varying degrees of precision, measure the same stars and save result
if generate_files:
    for error_perc in error_percs:
        logging.info("Generating files for {}% precision".format(error_perc))
        synth_tables[error_perc] =\
            syn.generate_table_with_error( sky_coord_now, 0.01*int(error_perc))
        with open(data_files[error_perc], 'w') as fp:
            pickle.dump(synth_tables[error_perc], fp)
        tb.traceback(synth_tables[error_perc],times,savefile=tb_files[error_perc])

# Plot the traceback alongside naiive volume measure

for error_perc in error_percs:
    logging.info("Clearing out temp_plots/ for all .png and .avi files")
    subprocess.call(['pwd'])
    subprocess.call(['rm', 'temp_plots/*.png'])
    subprocess.call(['rm', 'temp_plots/*.avi'])

    logging.info("{}% : calling ee.plot_something".format(error_perc))
    ee.plot_something((0,1), tb_files[error_perc], fit_bayes=fit_bayes)
    subprocess.call('./generate_synth_gif.sh')
    subprocess.call([
        'mv', 'temp_plots/video.avi',
        '../results/synth_gif_{}.avi'.format(error_perc)
    ])

