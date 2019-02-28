"""
Profile groupfitter

See where the bulk of computation occurs.

Examples on how to profile with python
https://docs.python.org/2/library/profile.html
"""

import cProfile
import logging
import numpy as np
import pstats
import sys

import chronostar.synthdata

sys.path.insert(0, '..')
import chronostar.synthdata as syn
import chronostar.measurer as ms
import chronostar.converter as cv
import chronostar.groupfitter as gf

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename='temp_logs/groupfitter.log')
    save_dir = 'temp_data/'
    group_savefile = save_dir + 'origins_stat.npy'
    xyzuvw_init_savefile = save_dir + 'xyzuvw_init_stat.npy'
    astro_savefile = save_dir + 'astro_table_stat.txt'
    xyzuvw_conv_savefile = save_dir + 'xyzuvw_conv_stat.fits'

    pars = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1e-8, 100])
    error_frac = 1
    xyzuvw_init, group = syn.synthesiseXYZUVW(pars, return_group=True,
                                              xyzuvw_savefile=xyzuvw_init_savefile,
                                              group_savefile=group_savefile,
                                              internal=True)
    astro_table = chronostar.synthdata.measureXYZUVW(xyzuvw_init, error_frac,
                                                     savefile=astro_savefile)
    star_pars = cv.convertMeasurementsToCartesian(astro_table,
                                                  savefile=xyzuvw_conv_savefile)

    stat_file = 'groupfitter.stat'
    # best_fit, chain, lnprob = \
    cProfile.run(
        "gf.fitGroup(xyzuvw_dict=star_pars, plot_it=True,"
        "convergence_tol=2., burnin_steps=400, plot_dir='temp_plots/',"
        "save_dir='temp_data/')",
        stat_file,
    )

    stat = pstats.Stats(stat_file)
    stat.sort('cumtime')
    stat.print_stats(0.1)




