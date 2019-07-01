
import numpy as np

assoc_name = 'new_synth_three_old'
config = {
    # 'datafile':'',
    'results_dir':'../results/{}'.format(assoc_name),
    #'data_loadfile':'../data/gaia_cartesian_full_6d_table.fits',
    'data_savefile':'../results/{}/{}_data.fit'.format(assoc_name, assoc_name),
    # 'datafile':'../results/{}/data.fits'.format(assoc_name),
    'plot_it':True,
    # 'background_overlaps_file':'',
    'include_background_distribution':False,
    # 'kernel_density_input_datafile':'../data/gaia_cartesian_full_6d_table.fits',
                                                    # Cartesian data of all Gaia DR2 stars
                                                    # e.g. ../data/gaia_dr2_mean_xyzuvw.npy
    'run_with_mpi':True,       # not yet inpmlemented
    'convert_to_cartesian':True,        # whehter need to convert data from astrometry to cartesian
    'overwrite_datafile':False,         # whether to store results in same talbe and rewrite to file
    'cartesian_savefile':'../results/{}/{}_data.fit'.format(assoc_name, assoc_name),
    'save_cartesian_data':True,         #
    'ncomps':1,                         # maximum number of components to reach
    'overwrite_prev_run':True,          # explores provided results directorty and sees if results already
                                        # exist, and if so picks up from where left off
    'dummy_trace_orbit_function':False,  # For testing, simple function to skip computation
    'pickup_prev_run':True,             # Pick up where left off if possible
    'banyan_assoc_name':'',
}

# synth = None
# parameters below correspond to current day means of:
# [80., -100., 35.,  4.1,  7.76, 4.25],
# [20.,  -80., 25., -1.9, 11.76, 2.25],
# [50., -100., 25.,  1.1,  7.76, 2.25],
synth = {
    'pars':np.array([
        [-264.6, -143.3, -67.9,  9.2, -1.6, -0.6, 10., 1.0, 30.],
        [-613.2,  -35.9,  -5.6, 15.6, -5.2, -2.8,  8., 1.5, 50.],
        [-576.4,  532.1,  39.4, 16.2, -9.9, -0.8,  6., 0.7, 70.],
    ]),
    'starcounts' : [50, 100, 40]
}

data_bound = None

historical_colnames = False

astro_colnames = {
    # 'main_colnames':None,     # list of names
    # 'error_colnames':None,
    # 'corr_colnames':None,
}

cart_colnames = {
    # 'main_colnames':None,
    # 'error_colnames':None,
    # 'corr_colnames':None,
}

special = {
    'component':'sphere',       # parameterisation for the origin
}

advanced = {
    'burnin_steps':500,        # emcee parameters, number of steps for each burnin iteraton
    'sampling_steps':500,
    'store_burnin_chains':False, # whether or not to store the sampling chains
}
