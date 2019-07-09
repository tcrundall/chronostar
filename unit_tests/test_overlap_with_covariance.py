#! /usr/bin/env python
"""
    author: Marusa Zerjal 2019 - 07 - 09
    Just playing...

    Small unit test used to...


"""
import logging
import numpy as np
import sys

sys.path.insert(0, '..')

from chronostar import expectmax
from chronostar.synthdata import SynthData
from chronostar.component import SphereComponent
from chronostar import tabletool

def test_compare_two_overlap_implementations():
    """
    author: Marusa Zerjal 2019 - 07 - 09

    Compare KDE and background overlap with covariance matrix

    """

    # Simple background data
    true_comp_mean = np.zeros(6)
    true_comp_dx = 200.
    true_comp_dv = 200.
    true_comp_covmatrix = np.identity(6)
    true_comp_covmatrix[:3, :3] *= true_comp_dx ** 2
    true_comp_covmatrix[3:, 3:] *= true_comp_dv ** 2
    true_comp_age = 1e-10
    true_comp = SphereComponent(attributes={
        'mean': true_comp_mean,
        'covmatrix': true_comp_covmatrix,
        'age': true_comp_age,
    })

    nstars = 2000
    synth_data = SynthData(pars=true_comp.get_pars(), starcounts=nstars)
    synth_data.synthesise_everything()
    tabletool.convert_table_astro2cart(synth_data.table)

    star_data = tabletool.build_data_dict_from_table(synth_data.table)
    background_means = star_data['means']

    # STAR DATA
    true_comp_mean = np.zeros(6)
    true_comp_dx = 2.
    true_comp_dv = 2.
    true_comp_covmatrix = np.identity(6)
    true_comp_covmatrix[:3, :3] *= true_comp_dx ** 2
    true_comp_covmatrix[3:, 3:] *= true_comp_dv ** 2
    true_comp_age = 1e-10
    true_comp = SphereComponent(attributes={
        'mean': true_comp_mean,
        'covmatrix': true_comp_covmatrix,
        'age': true_comp_age,
    })


    nstars = 10
    synth_data = SynthData(pars=true_comp.get_pars(), starcounts=nstars)
    synth_data.synthesise_everything()
    tabletool.convert_table_astro2cart(synth_data.table)

    star_data = tabletool.build_data_dict_from_table(synth_data.table)
    star_means = star_data['means']
    star_covs = star_data['covs']
    #group_mean = true_comp.get_mean()
    #group_cov = true_comp.get_covmatrix()

    # Background overlaps using KDE
    ln_bg_ols_kde = expectmax.get_kernel_densities(background_means, star_means, )
    #print(ln_bg_ols_kde)

    # Background overlaps using covariance matrix
    ln_bg_ols_cov = expectmax.get_background_overlaps_with_covariances(background_means, star_means, star_covs)
    #print(ln_bg_ols_cov)

    assert np.allclose(ln_bg_ols_kde, ln_bg_ols_cov, atol=0.1)

if __name__ == '__main__':
    test_compare_two_overlap_implementations()