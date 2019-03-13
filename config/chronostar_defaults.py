import numpy as np

synth_name = 'demo_run'
config = {
    # 'datafile':'',
    'results_dir':'../results/{}'.format(synth_name),
    'datafile':'../results/{}/data.fits'.format(synth_name),
    'plot_it':True,
    # 'background_overlaps_file':'',
    'include_background_distribution':True,
    'kernel_density_input_datafile':'',
    'run_with_mpi':False,
    'convert_to_cartesian':True,
    'overwrite_datafile':False,
    'cartesian_savefile':'',
    'save_cartesian_data':True,
    'ncomps':10,
    'overwrite_prev_run':True,
    'dummy_trace_orbit_function':True,
    'pickup_prev_run':True,
}

# synth = None
synth = {
    'pars':np.array([
        [ 50., 0.,10., 0., 0., 3., 5., 2., 1e-10],
        [-50., 0.,20., 0., 5., 2., 5., 2., 1e-10],
        [  0.,50.,30., 0., 0., 1., 5., 2., 1e-10],
    ]),
    'starcounts':[100,50,50]
}

astro_colnames = {
    # 'main_colnames':None,
    # 'error_colnames':None,
    # 'corr_colnames':None,
}

cart_colnames = {
    # 'main_colnames':None,
    # 'error_colnames':None,
    # 'corr_colnames':None,
}

special = {
    'component':'sphere',
}
