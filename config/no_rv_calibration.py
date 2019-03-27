import numpy as np

assoc_name = 'no_rv_calib'
config = {
    # 'datafile':'',
    'results_dir':'../results/{}'.format(assoc_name),
    #'data_loadfile': '../data/no_rv_paper/beta_Pictoris_with_gaia_small_everything_final.fits', # CHANGE RV UNCERTAINTIES HERE
    'data_loadfile': '../data/no_rv_paper/beta_Pictoris_with_gaia_small_everything_final_big_fake_rv_errors.fits', # CHANGE RV UNCERTAINTIES HERE
    'data_savefile': '../results/{}/beta_Pictoris_with_background.fits'.format(assoc_name), #,#''../results/{}/{}_subset.fit'.format(assoc_name, assoc_name), # Chronostar adds XYZUVW columns and
                                        # if you don't want to override the original file then save into data_savefile.
    'plot_it':True,
    # 'background_overlaps_file':'',
    'include_background_distribution':True,
    'kernel_density_input_datafile':'../data/gaia_cartesian_full_6d_table.fits',
                                                    # Cartesian data of all Gaia DR2 stars
                                                    # e.g. ../data/gaia_dr2_mean_xyzuvw.npy
    'run_with_mpi':False,       # not yet inpmlemented
    'convert_to_cartesian':True,        # whehter need to convert data from astrometry to cartesian
    'overwrite_datafile':False,         # whether to store results in same talbe and rewrite to file
    'cartesian_savefile':'',
    'save_cartesian_data':True,         #
    'ncomps':10,                        # maximum number of components to reach
    'overwrite_prev_run':True,          # explores provided results directorty and sees if results already
                                        # exist, and if so picks up from where left off
    'dummy_trace_orbit_function':True,  # For testing, simple function to skip computation
    'pickup_prev_run':True,             # Pick up where left off if possible
}

# dummy change

# synth = None
synth = {
   # 'pars':np.array([
   #     [ 50., 0.,10., 0., 0., 3., 5., 2., 1e-10],
   #     [-50., 0.,20., 0., 5., 2., 5., 2., 1e-10],
   #     [  0.,50.,30., 0., 0., 1., 5., 2., 1e-10],
   # ]),
   # 'starcounts':[100,50,50]
}

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
    'burnin_steps':1000,        # emcee parameters, number of steps for each burnin iteraton
    'sampling_steps':1000,
}
