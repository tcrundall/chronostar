from __future__ import print_function, division

import numpy as np
import pdb
import sys
sys.path.insert(0,'..')

import chronostar.synthesiser as syn
import chronostar.measurer as ms
import chronostar.converter as cv
import chronostar.groupfitter as gf
import chronostar.expectmax as em


def test_calcMedAndSpan():
    """
    Test that the median, and +- 34th percentiles is found correctly
    """
    dx = 10.
    dv = 5.
    dummy_mean = np.array([10,10,10, 5, 5, 5,np.log(dx),np.log(dv),20])
    dummy_std =  np.array([1.,1.,1.,1.,1.,1.,0.5,       0.5,       3.])
    assert len(dummy_mean) == len(dummy_std)
    npars = len(dummy_mean)

    nsteps = 10000
    nwalkers = 18

    dummy_chain = np.array([np.random.randn(nsteps)*std + mean
                            for (std, mean) in zip(dummy_std, dummy_mean)]).T
    np.repeat(dummy_chain, 18, axis=0).reshape(nwalkers,nsteps,npars)

    med_and_span = em.calcMedAndSpan(dummy_chain)
    assert np.allclose(dummy_mean, med_and_span[:,0], atol=0.1)
    approx_stds = 0.5*(med_and_span[:,1] - med_and_span[:,2])
    assert np.allclose(dummy_std, approx_stds, atol=0.1)

def test_calcMembershipProbs():
    """
    Even basicer. Checks that differing overlaps are
    correctly mapped to memberships.
    """

    # case 1
    star_ols = [10, 10]
    assert np.allclose([.5,.5], em.calcMembershipProbs(np.log(star_ols)))

    # case 2
    star_ols = [10, 30]
    assert np.allclose([.25,.75], em.calcMembershipProbs(np.log(star_ols)))

    # case 3
    star_ols = [10, 10, 20]
    assert np.allclose([.25, .25, .5],
                       em.calcMembershipProbs(np.log(star_ols)))


def test_expectation():
    """
    Super basic, generates some association stars along
    with some background stars and checks membership allocation
    is correct
    """
    nfield = 10
    nass_stars = 20
    nstars = nfield + nass_stars

    old_z = np.zeros((nstars, 2))
    old_z[:nass_stars, 0] = 1.
    old_z[nass_stars:, 1] = 1.

    age = 1e-5
    ass_pars = np.array([0, 0, 0, 0, 0, 0, 5., 2., age, nass_stars])
    ass_xyzuvw_init, ass_group =\
        syn.synthesiseXYZUVW(ass_pars, return_group=True, internal=False)
    field_xyzuvw_init = np.random.uniform(-15, 15, (nfield, 6))

    all_xyzuvw_init = np.vstack((ass_xyzuvw_init, field_xyzuvw_init))

    astro_table = ms.measureXYZUVW(all_xyzuvw_init, error_frac=1.0)
    star_pars = cv.convertMeasurementsToCartesian(astro_table)
    lnols = np.log(nass_stars) +\
            gf.getLogOverlaps(ass_group.getInternalSphericalPars(), star_pars)
    bg_ln_ols = np.array([-34] * nstars)

    z = em.expectation(star_pars, [ass_group], bg_ln_ols=bg_ln_ols,
                       old_z=old_z)
    strong_bg_overlap = np.where(lnols < bg_ln_ols)

    assert (z[strong_bg_overlap, 1] > 0.5).all()

