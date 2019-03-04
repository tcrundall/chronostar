"""
Test various capabilities of the Component class
"""
from __future__ import print_function, division, unicode_literals

import numpy as np
import sys
sys.path.insert(0, '..')
from chronostar.component import Component

DX = 10.
DV = 5.
AGE = 10
SPHERE_PARS = np.array([0., 0., 0., 0., 0., 0., DX, DV, AGE])
COMP_FORM = 'sphere'

def test_initialisation():
    """Checks objects are reasonably initiated"""
    mycomp = Component(pars=SPHERE_PARS, form=COMP_FORM)
    assert np.allclose(SPHERE_PARS, mycomp.pars)


def test_externalisePars():
    """Check that pars are successfully converted from internal form (used by
    emcee) to external form (interacted with by user) successfully"""
    internal_pars = np.copy(SPHERE_PARS)
    internal_pars[6:8] = np.log(internal_pars[6:8])
    external_pars = Component.externalisePars(internal_pars, form=COMP_FORM)
    assert np.allclose(SPHERE_PARS, external_pars)


def test_formConversions():
    """
    Confirms the following:
     - sphere components generate elliptical parametrisation correctly
     - sphere and ellipitcal components generate same mean and covariance
       matrix when initialised with equivalent parameters
    """
    sphere_comp = Component(pars=SPHERE_PARS, form='sphere')
    elliptical_pars = sphere_comp.getEllipticalPars()
    elliptical_comp = Component(pars=elliptical_pars, form='elliptical')

    assert np.allclose(sphere_comp.mean, elliptical_comp.mean)
    assert np.allclose(sphere_comp.dx, elliptical_comp.sphere_dx)
    assert np.allclose(sphere_comp.generateCovMatrix(),
                       elliptical_comp.generateCovMatrix())


def test_simpleProjection():
    """
    Check negligble change in mean and covmatrix when projected for negligible
    timestep
    """
    tiny_age = 1e-10
    tiny_age_pars = np.copy(SPHERE_PARS)
    tiny_age_pars[-1] = tiny_age

    comp = Component(pars=tiny_age_pars, form='sphere')
    comp.calcCurrentDayProjection()

    assert np.allclose(comp.mean, comp.mean_now, atol=1e-10)
    assert np.allclose(comp.covmatrix, comp.covmatrix_now, atol=1e-8),\
        'comparing\n{}\nwith\n{}'.format(comp.covmatrix, comp.covmatrix_now)


def test_splitGroup():
    """
    Splitting group by provided ages yields identical initial cov matrix,
    and identical current day mean
    """
    comp = Component(pars=SPHERE_PARS, form='sphere')
    offset = 5.
    assert offset < comp.age        #check that we won't get negative ages
    lo_age, hi_age = comp.age - offset, comp.age + offset
    lo_comp, hi_comp = comp.splitGroup(lo_age, hi_age)

    comp.calcCurrentDayProjection()
    lo_comp.calcCurrentDayProjection()
    hi_comp.calcCurrentDayProjection()

    assert lo_age == lo_comp.age
    assert hi_age == hi_comp.age
    assert np.allclose(comp.covmatrix, lo_comp.covmatrix)
    assert np.allclose(comp.covmatrix, hi_comp.covmatrix)

    assert np.allclose(comp.mean_now, lo_comp.mean_now, atol=1e-10)
    assert np.allclose(comp.mean_now, hi_comp.mean_now, atol=1e-10)

    assert np.allclose(lo_comp.mean, lo_comp.pars[:6])


def test_loadComponents():
    single_filename = 'temp_data/single_comp.npy'
    multi_filename = 'temp_data/multi_comp.npy'

    comp0 = Component(pars=SPHERE_PARS)
    comp1 = Component(pars=SPHERE_PARS)

    np.save(single_filename, comp0)
    np.save(multi_filename, [comp0, comp1])

    single_res = Component.loadComponents(single_filename)
    # assert single_res[0].pars == comp1.pars

    multi_res = Component.loadComponents(multi_filename)
    assert np.allclose(multi_res[0].pars, comp0.pars)
    assert np.allclose(multi_res[1].pars, comp1.pars)


def test_setParsFromMeanAndCov():
    sphere_pars = np.copy(SPHERE_PARS)
    sphere_pars[-1] = Component.DEFAULT_TINY_AGE
    comp_orig = Component(pars=sphere_pars)

    orig_mean = comp_orig.mean
    orig_covmatrix = comp_orig.covmatrix

    comp_from_mc = Component(mean=orig_mean, covmatrix=orig_covmatrix)

    assert np.allclose(comp_orig.pars, comp_from_mc.pars)

    ellip_pars = np.array([0.,0.,0.,0.,0.,0.,
                           16.,9.,9.,2.,0.5,0.2,-0.4,
                           Component.DEFAULT_TINY_AGE])
    ellip_comp_orig = Component(pars=ellip_pars, form='elliptical')
    ellip_orig_mean = ellip_comp_orig.mean
    ellip_orig_covmatrix = ellip_comp_orig.covmatrix

    ellip_comp_from_mc = Component(mean=ellip_orig_mean,
                                   covmatrix=ellip_orig_covmatrix,
                                   form='elliptical')
    assert np.allclose(ellip_comp_orig.pars, ellip_comp_from_mc.pars)
    assert np.isclose(ellip_comp_orig.sphere_dx, ellip_comp_from_mc.sphere_dx)

if __name__ == '__main__':
    pass
