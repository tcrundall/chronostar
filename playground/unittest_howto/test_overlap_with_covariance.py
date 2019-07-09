#! /usr/bin/env python
"""
    author: Marusa Zerjal 2019 - 07 - 09
    Just playing...

    Small unit test used to...


"""
import logging
import numpy as np
import sys

sys.path.insert(0, '../../')

from chronostar import expectmax
from chronostar.synthdata import SynthData
from chronostar.component import SphereComponent
from chronostar import tabletool

def test_compare_two_overlap_implementations():
    """
    author: Marusa Zerjal 2019 - 07 - 09

    Compare KDE and background overlap with covariance matrix

    """

    # Simple data
    background_means = np.array(
        [[-460, 509, -37, 311, -364, 99],
        [402, 553, -103, -247, -353, 79],
        [-78, 374, -446, 177, -204, 381],
        [-239,  452, 24, 173, -398, 2],
        [-94, 191, -100, 290, -317, 153],
        [239, 409, -371, -116, -361, 196],
        [-1, -26, 61, 57, 256, -330],
        [-98, 344, 184, 387, -290, -134],
        [226, 116, 160, -361, -267, -16],
        [-259, 828, 200, 79, -389, -91]]
    )


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


    nstars = 2
    synth_data = SynthData(pars=true_comp.get_pars(), starcounts=nstars)
    synth_data.synthesise_everything()
    tabletool.convert_table_astro2cart(synth_data.table)

    star_data = tabletool.build_data_dict_from_table(synth_data.table)
    star_means = star_data['means']
    star_covs = star_data['covs']
    #group_mean = true_comp.get_mean()
    #group_cov = true_comp.get_covmatrix()

    print('Start')

    # Background overlaps using KDE
    ln_bg_ols_kde = expectmax.get_kernel_densities(background_means, star_means, )
    print('KDE finished.')
    print(ln_bg_ols_kde)

    # Background overlaps using covariance matrix
    print('Start bg ols cov')
    ln_bg_ols_cov = expectmax.get_background_overlaps_with_covariances(background_means, star_means, star_covs)
    print('BG OLS COV finished.')
    print(ln_bg_ols_cov)

    assert np.allclose(ln_bg_ols_kde, ln_bg_ols_cov)

if __name__ == '__main__':
    test_compare_two_overlap_implementations()