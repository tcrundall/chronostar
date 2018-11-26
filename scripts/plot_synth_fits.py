from __future__ import print_function, division

"""
Plot a bunch of synth fits
"""

from itertools import product
import numpy as np
import os
import sys
from shutil import copyfile

sys.path.insert(0, '..')

import chronostar.datatool as dt
import chronostar.synthesiser as syn
import chronostar.fitplotter as fp

ages = [5,15,30,50,100,200]
dxs = [1, 2]
dvs = [1, 2]
nstars = [25, 50, 100]
labels = 'abcd'
precs = ['half', 'gaia', 'double', 'quint']
#
#prob_scenarios =\
#    [[5, 1, 2, 50, 'd', 'half'],
#     [30, 1, 2, 50, 'a', 'gaia'],
#     [30, 2, 2, 25, 'd', 'half'],
#     [30, 2, 2, 25, 'd', 'gaia'],
#     [30, 2, 2, 25, 'd', 'double'],
#     [30, 2, 2, 100, 'a', 'half'],
#     [30, 2, 2, 100, 'a', 'gaia'],
#     [30, 2, 2, 100, 'a', 'double'],
#     [50, 1, 2, 100, 'c', 'half'],
#     [50, 1, 2, 100, 'c', 'gaia'],
#     [50, 2, 2, 25, 'c', 'half'],
#     [50, 2, 2, 25, 'c', 'gaia'],
#     [50, 2, 2, 25, 'c', 'double'],
#     [50, 2, 2, 100, 'd', 'half']
#     ]
#
#worst_raw_scenarios =\
#[[5, 1, 1, 25, 'd', 'double'],
# [5, 1, 1, 100, 'b', 'double'],
# [5, 2, 1, 25, 'b', 'half'],
# [5, 2, 1, 25, 'b', 'gaia'],
# [5, 2, 1, 25, 'b', 'double'],
# [5, 2, 1, 50, 'a', 'gaia'],
# [5, 2, 1, 50, 'c', 'gaia'],
# [5, 2, 1, 100, 'd', 'half'],
# [5, 2, 1, 100, 'd', 'gaia'],
# [5, 2, 1, 100, 'd', 'double'],
# [15, 1, 1, 25, 'd', 'double'],
# [15, 1, 1, 50, 'b', 'gaia'],
# [15, 1, 1, 50, 'b', 'double'],
# [15, 1, 1, 50, 'c', 'double'],
# [15, 1, 1, 100, 'a', 'double'],
# [15, 1, 1, 100, 'b', 'double'],
# [15, 1, 1, 100, 'c', 'double'],
# [15, 1, 1, 100, 'd', 'double'],
# [15, 1, 2, 25, 'd', 'double'],
# [15, 1, 2, 50, 'a', 'double'],
# [15, 1, 2, 50, 'c', 'double'],
# [15, 2, 1, 25, 'a', 'half'],
# [15, 2, 1, 25, 'a', 'gaia'],
# [15, 2, 1, 25, 'c', 'half'],
# [15, 2, 1, 25, 'c', 'gaia'],
# [15, 2, 1, 25, 'c', 'double'],
# [15, 2, 1, 50, 'a', 'half'],
# [15, 2, 1, 50, 'a', 'gaia'],
# [15, 2, 1, 50, 'a', 'double'],
# [15, 2, 1, 50, 'b', 'gaia'],
# [15, 2, 1, 50, 'b', 'double'],
# [15, 2, 1, 50, 'c', 'double'],
# [15, 2, 1, 50, 'd', 'double'],
# [15, 2, 1, 100, 'a', 'gaia'],
# [15, 2, 1, 100, 'a', 'double'],
# [15, 2, 1, 100, 'c', 'double'],
# [15, 2, 1, 100, 'd', 'double'],
# [15, 2, 2, 50, 'b', 'half'],
# [15, 2, 2, 50, 'b', 'gaia'],
# [15, 2, 2, 50, 'b', 'double'],
# [15, 2, 2, 50, 'd', 'double'],
# [15, 2, 2, 100, 'c', 'double'],
# [30, 1, 1, 25, 'b', 'gaia'],
# [30, 1, 1, 25, 'd', 'half'],
# [30, 1, 1, 25, 'd', 'gaia'],
# [30, 1, 1, 50, 'a', 'gaia'],
# [30, 1, 1, 50, 'a', 'double'],
# [30, 2, 1, 25, 'b', 'half'],
# [30, 2, 1, 25, 'b', 'gaia'],
# [30, 2, 1, 25, 'b', 'double'],
# [30, 2, 1, 50, 'a', 'half'],
# [30, 2, 1, 50, 'a', 'gaia'],
# [30, 2, 1, 50, 'a', 'double'],
# [30, 2, 1, 50, 'b', 'gaia'],
# [30, 2, 1, 50, 'd', 'double'],
# [30, 2, 1, 100, 'b', 'double'],
# [30, 2, 2, 25, 'd', 'half'],
# [30, 2, 2, 25, 'd', 'gaia'],
# [30, 2, 2, 25, 'd', 'double'],
# [30, 2, 2, 100, 'a', 'half'],
# [30, 2, 2, 100, 'a', 'gaia'],
# [30, 2, 2, 100, 'a', 'double'],
# [50, 1, 1, 25, 'd', 'half'],
# [50, 1, 1, 25, 'd', 'gaia'],
# [50, 1, 1, 25, 'd', 'double'],
# [50, 1, 1, 100, 'a', 'double'],
# [50, 1, 1, 100, 'd', 'double'],
# [50, 1, 2, 25, 'a', 'gaia'],
# [50, 1, 2, 100, 'c', 'half'],
# [50, 1, 2, 100, 'c', 'gaia'],
# [50, 2, 1, 50, 'a', 'gaia'],
# [50, 2, 1, 50, 'a', 'double'],
# [50, 2, 1, 50, 'b', 'half'],
# [50, 2, 1, 50, 'b', 'gaia'],
# [50, 2, 1, 50, 'c', 'double'],
# [50, 2, 2, 25, 'a', 'gaia'],
# [50, 2, 2, 25, 'c', 'half'],
# [50, 2, 2, 25, 'c', 'gaia'],
# [50, 2, 2, 25, 'c', 'double'],
# [50, 2, 2, 25, 'd', 'half'],
# [50, 2, 2, 25, 'd', 'gaia'],
# [50, 2, 2, 25, 'd', 'double'],
# [50, 2, 2, 50, 'c', 'gaia'],
# [50, 2, 2, 50, 'c', 'double'],
# [50, 2, 2, 100, 'b', 'double'],
# [50, 2, 2, 100, 'd', 'half'],
# [50, 2, 2, 100, 'd', 'gaia']]
#

MASTER_DIR = '/data/mash/tcrun/synth_fit/'
if not os.path.exists(MASTER_DIR):
    MASTER_DIR = '../results/synth_fit/'
    

for scenario in product(ages, dxs, dvs, nstars, labels, precs):
    rdir = MASTER_DIR+'{}_{}_{}_{}_{}/{}/'.format(*scenario)

    plt_file = rdir + 'multi_plot_{}_{}_{}_{}_{}_{}.pdf'.format(*scenario)
    print("Checking {}".format(plt_file))
    if not os.path.isfile(plt_file):
        print("Plotting {}".format(plt_file))
        try:
            star_pars_file = rdir + 'xyzuvw_now.fits'
            chain_file = rdir + 'final_chain.npy'
            origins_file = rdir + 'origins.npy'
            lnprob_file = rdir + 'final_lnprob.npy'

            chain = np.load(chain_file).reshape(-1,9)
            lnprob = np.load(lnprob_file)
            best_pars = chain[np.argmax(lnprob_file)]
            best_fit = syn.Group(best_pars, internal=True, starcount=False)
            origins = dt.loadGroups(origins_file)

            star_pars = dt.loadXYZUVW(star_pars_file)

            fp.plotMultiPane(
                ['xy', 'xz', 'uv', 'xu', 'yv', 'zw'],
                star_pars,
                [best_fit],
                origins=origins,
                save_file=rdir+'multi_plot_{}_{}_{}_{}_{}_{}.pdf'.format(*scenario),
                title='{}Myr, {}pc, {}km/s, {} stars, {}, {}'.format(*scenario),
            )
            print("done")
        except:
            print("Not ready yet...")
        

#
#prob_dir = '../results/problem_paper1_runs/'
#worst_raw_dir = '../results/problem_raw_paper1_runs/'
#try:
#    os.mkdir(worst_raw_dir)
#except OSError:
#    print("Directory already exists... carrying on")
#for scenario in worst_raw_scenarios: #prob_scenarios:
#    rdir = '../results/paper1_runs/{}_{}_{}_{}_{}/{}/'.format(*scenario)
#    plt_file = 'multi_plot_{}_{}_{}_{}_{}_{}.pdf'.format(*scenario)
#    plt_path = rdir + plt_file
#    copyfile(plt_path, worst_raw_dir + plt_file)
