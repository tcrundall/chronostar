import numpy as np

assoc_name = ''
prefix = 'tweak_galpy_potential_mw_tims'
config = {
    # 'datafile':'',
    'results_dir':'../results/{}'.format(prefix),
    'data_loadfile': '../data/same_centroid_synth_measurements.fits',
    'data_savefile': '../results/{}/same_centroid_synth_measurements_output.fits'.format(prefix), #,#''../results/{}/{}_subset.fit'.format(assoc_name, assoc_name), # Chronostar adds XYZUVW columns and
                                        # if you don't want to override the original file then save into data_savefile.
    'plot_it':True, # What kind of plots?
    # 'background_overlaps_file':'',
    'include_background_distribution':False, # Synthetic data doesnt have background component
    'kernel_density_input_datafile':None, # '/home/tcrun/chronostar/data/gaia_cartesian_full_6d_table.fits', # absolute path as I ran out of space
                                                    # Cartesian data of all Gaia DR2 stars
                                                    # e.g. ../data/gaia_dr2_mean_xyzuvw.npy
    'run_with_mpi':True,       # not yet inpmlemented
    'convert_to_cartesian':True,        # whehter need to convert data from astrometry to cartesian
    'overwrite_datafile':False,         # whether to store results in same talbe and rewrite to file
    # 'cartesian_savefile':'../results/{}/same_centroid_synth_measurements_with_xyzuvw.fits'.format(prefix),
    'save_cartesian_data':True,         #
    'overwrite_prev_run':True,          # explores provided results directorty and sees if results already
                                        # exist, and if so picks up from where left off
    'dummy_trace_orbit_function':False,  # For testing, simple function to skip computation
    'pickup_prev_run':True,             # Pick up where left off if possible
    'banyan_assoc_name': '',
}

# synth = None
synth = {
   # 'pars':np.array([
   #     [ 50., 0.,10., 0., 0., 3., 5., 2., 1e-10],
   #     [-50., 0.,20., 0., 5., 2., 5., 2., 1e-10],
   #     [  0.,50.,30., 0., 0., 1., 5., 2., 1e-10],
   # ]),
   # 'starcounts':[100,50,50]
}

data_bound = None

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

from galpy.potential import MiyamotoNagaiPotential, MWPotential2014
orbit = {
    # 'potential': None,
    'potential': MWPotential2014,
    #'potential': MiyamotoNagaiPotential(a=0.5,b=0.0375,amp=1.,normalize=1.), # Params from the example webpage. No idea if that's good or not.
}

special = {
    'component':'sphere',       # parameterisation for the origin
}

advanced = {
    'burnin_steps':500,        # emcee parameters, number of steps for each burnin iteraton
    'sampling_steps':500,
}
