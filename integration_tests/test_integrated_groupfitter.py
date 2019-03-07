from __future__ import print_function, division, unicode_literals

import logging
import numpy as np
import sys

sys.path.insert(0,'..')
from chronostar.component import SphereComponent
from chronostar.synthdata import SynthData
from chronostar import tabletool
import chronostar.groupfitter as gf
# import chronostar.converter as cv
# import chronostar.measurer as ms
# import chronostar.synthdata as syn

def test_stationary_component():
    """
    Integrated test which fits a single component to a synthetic association.

    Runtime on my mac (single thread) is ~ 20 mins. Check logs/groupfitter.log
    and temp_plots/*.png for progress.

    With generous convergence, takes about 5 mins single thread.
    """
    logging.basicConfig(level=logging.INFO, filename='logs/groupfitter.log',
                        filemode='w')
    generous_convergence_tol = 1.
    short_burnin_step = 200
    save_dir = 'temp_data/'
    plot_dir = 'temp_plots/'
    group_savefile = save_dir + 'origins_stat.npy'
    synth_data_savefile = 'temp_data/groupfitter_synthdata.fits'

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
    synth_data = SynthData(pars=true_comp.get_pars(),
                           starcounts=nstars,
                           measurement_error=measurement_error)
    synth_data.synthesiseEverything()
    tabletool.convertTableAstroToXYZUVW(synth_data.astr_table,
                                        write_table=True,
                                        filename=synth_data_savefile)

    best_comp, chain, lnprob = gf.fit_group(
            data=synth_data.astr_table,
            plot_it=True,
            convergence_tol=generous_convergence_tol,
            burnin_steps=short_burnin_step,
            plot_dir=plot_dir,
            save_dir=save_dir,
    )

    return true_comp, best_comp

    assert np.allclose(true_comp.get_mean(), best_comp.get_mean(),
                       atol=1.0)
    assert np.allclose(true_comp.get_age(), best_comp.get_age(),
                       atol=0.5)
    assert np.allclose(true_comp.get_covmatrix(),
                       best_comp.get_covmatrix(),
                       atol=2.0)

if __name__ == '__main__':
    true_comp, best_comp = test_stationary_component()
