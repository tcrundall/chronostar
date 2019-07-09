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

    Generate two sets of normally distributed points. The first serves
    as the background stars, the second is our set of sample stars
    for which the background overlaps will be calculated. In the limit
    of tiny covariance matrices for the stars,
    """
    TINY_STD = 1e-6

    # Simple background data
    nbgstars = 1000
    bg_comp_dx = 100.
    bg_comp_dv = 10.
    background_means = np.random.randn(nbgstars, 6)
    background_means[:, :3] *= bg_comp_dx
    background_means[:, 3:] *= bg_comp_dv

    # STAR DATA
    n_sample_stars = 10

    # Generating random values normally distributed
    star_means = np.random.randn(n_sample_stars, 6)

    # Extending spread to match that of background stars
    star_means[:,:3] *= bg_comp_dx
    star_means[:,3:] *= bg_comp_dv

    star_covs = np.array(
        n_sample_stars * [np.identity(6) * TINY_STD**2]
    )

    # Background overlaps using KDE
    ln_bg_ols_kde = expectmax.get_kernel_densities(background_means, star_means)

    # Background overlaps using covariance matrix
    ln_bg_ols_cov = expectmax.get_background_overlaps_with_covariances(background_means, star_means, star_covs)

    assert np.allclose(ln_bg_ols_kde, ln_bg_ols_cov)
    assert np.isfinite(ln_bg_ols_kde).all()
    assert np.isfinite(ln_bg_ols_cov).all()

