from __future__ import print_function, division, unicode_literals

import logging
import numpy as np
import sys

sys.path.insert(0,'..')
from chronostar.component import SphereComponent
from chronostar.synthdata import SynthData
from chronostar.traceorbit import traceOrbitXYZUVW
from chronostar import tabletool
import chronostar.groupfitter as gf

PY_VERS = sys.version[0]

def run_fit_helper(true_comp, starcounts, measurement_error,
                   burnin_step=None,
                   run_name='default',
                   ):
    py_vers = sys.version[0]
    data_filename = 'temp_data/{}_groupfitter_{}.fits'.format(py_vers, run_name)
    log_filename = 'logs/{}_groupfitter_{}.log'.format(py_vers, run_name)
    plot_dir = 'temp_plots/{}_groupfitter_{}'.format(py_vers, run_name)
    save_dir = 'temp_data/'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='w')
    synth_data = SynthData(pars=true_comp.get_pars(),
                           starcounts=starcounts,
                           measurement_error=measurement_error)
    synth_data.synthesise_everything()
    tabletool.convertTableAstroToXYZUVW(synth_data.table,
                                        write_table=True,
                                        filename=data_filename)
    res = gf.fit_comp(
            data=synth_data.table,
            plot_it=True,
            burnin_steps=burnin_step,
            plot_dir=plot_dir,
            save_dir=save_dir,
    )
    return res

def test_stationary_component():
    """
    Integrated test which fits a single component to a synthetic association.

    Runtime on my mac (single thread) is ~ 20 mins. Check logs/groupfitter.log
    and temp_plots/*.png for progress.

    Takes about 10 mins single thread with C implementation of overlap
    or ~40 mins with python implementation of overlap
    """
    # log_filename = 'logs/groupfitter_stationary.log'
    # synth_data_savefile = 'temp_data/groupfitter_stationary_synthdata.fits'

    short_burnin_step = 200

    true_comp_mean = np.zeros(6)
    true_comp_dx = 2.
    true_comp_dv = 2.
    true_comp_covmatrix = np.identity(6)
    true_comp_covmatrix[:3,:3] *= true_comp_dx**2
    true_comp_covmatrix[3:,3:] *= true_comp_dv**2
    true_comp_age = 1e-10
    true_comp = SphereComponent(attributes={
        'mean':true_comp_mean,
        'covmatrix':true_comp_covmatrix,
        'age':true_comp_age,
    })
    nstars = 100
    measurement_error = 1e-10

    best_comp, chain, lnprob = run_fit_helper(
            true_comp=true_comp, starcounts=nstars,
            measurement_error=measurement_error,
            run_name='stationary',
            burnin_step=short_burnin_step,
    )
    np.save('temp_data/{}_groupfitter_stationary_' \
            'true_and_best_comp.npy'.format(PY_VERS),
            [true_comp, best_comp],)


    assert np.allclose(true_comp.get_mean(), best_comp.get_mean(),
                       atol=1.0)
    assert np.allclose(true_comp.get_age(), best_comp.get_age(),
                       atol=0.5)
    assert np.allclose(true_comp.get_covmatrix(),
                       best_comp.get_covmatrix(),
                       atol=2.0)

def test_lcc_like():
    """
    Takes about 40 mins
    """
    mean_now = np.array([50., -100., 25., 1.1, -7.76, 2.25])

    age = 10.
    mean = traceOrbitXYZUVW(mean_now, times=-age)
    dx = 5.
    dv = 2.
    covmatrix = np.identity(6)
    covmatrix[:3,:3] *= dx**2
    covmatrix[3:,3:] *= dv**2

    true_comp = SphereComponent(attributes={
        'mean':mean,
        'covmatrix':covmatrix,
        'age':age,
    })

    nstars = 100
    tiny_measurement_error = 1e-10
    short_burnin_step = 200

    best_comp, chain, lnprob = run_fit_helper(
            true_comp=true_comp, starcounts=nstars,
            measurement_error=tiny_measurement_error,
            burnin_step=short_burnin_step,
            run_name='lcc_like',
    )

    np.save('temp_data/{}_groupfitter_lcc_like_'\
            'true_and_best_comp.npy'.format(PY_VERS),
            [true_comp, best_comp],)

    assert np.allclose(true_comp.get_mean(), best_comp.get_mean(),
                       atol=3.0)
    assert np.allclose(true_comp.get_age(), best_comp.get_age(),
                       atol=1.0)
    assert np.allclose(true_comp.get_covmatrix(),
                       best_comp.get_covmatrix(),
                       atol=5.0)

if __name__ == '__main__':
    true_comp, best_comp = test_stationary_component()
