"""
Tests the inheritance class structure of various
component models
If a new concrete component class has been implemented, simply
import, include in COMPONENT_CLASSES dictionary, and provide
some default parameters in DEFAULT_PARS, and it will
automatically be incorporated into tests, ensuring viable
interface has been implemented correctly.
"""

from __future__ import print_function, division, unicode_literals

import numpy as np

import sys
sys.path.insert(0, '..')

from chronostar.component import SphereComponent, EllipComponent

COMPONENT_CLASSES = {
    'sphere':SphereComponent,
    'ellip':EllipComponent,
    # Insert new implementations here
}

MEAN = np.array([0.,1.,2.,3.,4.,5.])
DX = 9.
DY = 12.
DZ = 16.
DV = 5.
C_XY = 0.2
C_XZ = -0.9
C_YZ = -0.3
AGE = 20
SPHERE_PARS = np.hstack((MEAN, [DX, DV, AGE]))

ELLIP_PARS = np.hstack((MEAN, [DX, DY, DZ, DV, C_XY, C_XZ, C_YZ, AGE]))

DEFAULT_PARS = {
    'sphere':SPHERE_PARS,
    'ellip':ELLIP_PARS,
    # Insert new default pars here
}


def test_general_initialisation():
    for name, ComponentClass in COMPONENT_CLASSES.items():
        comp = ComponentClass(pars=DEFAULT_PARS[name])
        assert np.allclose(MEAN, comp.get_mean())
        assert AGE == comp.get_age()


def test_spherecomponent_initialisation():
    sphere_comp = SphereComponent(pars=SPHERE_PARS, internal=False)
    assert np.allclose(SPHERE_PARS[:6], sphere_comp._mean)
    assert np.allclose(AGE, sphere_comp._age)
    assert np.isclose(DX, sphere_comp.get_sphere_dx())
    assert np.isclose(DV, sphere_comp.get_sphere_dv())


def test_ellipcomponent_initialisation():
    ellip_pars = np.copy(ELLIP_PARS)
    # remove correlations for sphere_dx checks
    ellip_pars[10:13] = 0.

    ellip_comp = EllipComponent(pars=ellip_pars, internal=False)
    assert np.allclose(ellip_pars[:6], ellip_comp._mean)
    assert np.allclose(AGE, ellip_comp._age)

    sphere_dx = (DX * DY * DZ)**(1./3)
    assert np.isclose(sphere_dx, ellip_comp.get_sphere_dx())
    assert np.isclose(DV, ellip_comp.get_sphere_dv())


def test_generic_externalise_and_internalise():
    for name, ComponentClass in COMPONENT_CLASSES.items():
        comp = ComponentClass(pars=DEFAULT_PARS[name], internal=False)

        internal_pars = comp.internalise(comp.get_pars())
        external_pars = comp.externalise(internal_pars)

        assert np.allclose(comp.get_pars(), external_pars)


def test_externalise_and_internalise_pars():
    """Check that pars are successfully converted from internal form (used by
    emcee) to external form (interacted with by user) successfully"""

    # Check SphereComponent
    internal_sphere_pars = np.copy(SPHERE_PARS)
    internal_sphere_pars[6:8] = np.log(internal_sphere_pars[6:8])
    external_sphere_pars = SphereComponent.externalise(internal_sphere_pars)
    assert np.allclose(SPHERE_PARS, external_sphere_pars)

    re_internal_sphere_pars = SphereComponent.internalise(external_sphere_pars)
    assert np.allclose(internal_sphere_pars, re_internal_sphere_pars)

    # Check EllipComponent
    internal_ellip_pars = np.copy(ELLIP_PARS)
    internal_ellip_pars[6:10] = np.log(internal_ellip_pars[6:10])
    external_ellip_pars = EllipComponent.externalise(internal_ellip_pars)
    assert np.allclose(ELLIP_PARS, external_ellip_pars)

    re_internal_ellip_pars = EllipComponent.internalise(external_ellip_pars)
    assert np.allclose(internal_ellip_pars, re_internal_ellip_pars)


def test_simple_projection():
    """
    Check negligible change in mean and covmatrix when projected for negligible
    timestep
    """
    tiny_age = 1e-10
    for name, ComponentClass in COMPONENT_CLASSES.items():
        comp = ComponentClass(pars=DEFAULT_PARS[name])
        comp.update_attribute(attributes={'age':tiny_age})

        assert np.allclose(comp.get_mean(), comp.get_mean_now(), atol=1e-8)
        assert np.allclose(comp.get_covmatrix(), comp.get_covmatrix_now(),
                           atol=1e-6)

def test_split_group():
    """
    Splitting group by provided ages yields identical initial cov matrix,
    and identical current day mean
    """
    for name, ComponentClass in COMPONENT_CLASSES.items():
        comp = ComponentClass(pars=DEFAULT_PARS[name])
        age_offset = 1.
        assert age_offset < comp._age    #check that we won't get negative ages
        lo_age, hi_age = comp._age - age_offset, comp._age + age_offset
        lo_comp, hi_comp = comp.splitGroup(lo_age, hi_age)

        assert lo_age == lo_comp.get_age()
        assert hi_age == hi_comp.get_age()
        assert np.allclose(comp.get_covmatrix(), lo_comp.get_covmatrix())
        assert np.allclose(comp.get_covmatrix(), hi_comp.get_covmatrix())

        assert np.allclose(comp.get_mean_now(), lo_comp.get_mean_now(),
                           atol=1e-4)
        assert np.allclose(comp.get_mean_now(), hi_comp.get_mean_now(),
                           atol=1e-4)


def test_load_components():
    single_filename = 'temp_data/single_comp.npy'
    multi_filename = 'temp_data/multi_comp.npy'
    for name, ComponentClass in COMPONENT_CLASSES.items():
        comp0 = ComponentClass(pars=DEFAULT_PARS[name])
        comp1 = ComponentClass(pars=DEFAULT_PARS[name])

        np.save(single_filename, comp0)
        np.save(multi_filename, [comp0, comp1])

        single_res = ComponentClass.load_components(single_filename)
        assert np.allclose(single_res[0].get_pars(), comp1.get_pars())

        multi_res = ComponentClass.load_components(multi_filename)
        assert np.allclose(multi_res[0].get_pars(), comp0.get_pars())
        assert np.allclose(multi_res[1].get_pars(), comp1.get_pars())


def test_init_from_attributes():
    for name, ComponentClass in COMPONENT_CLASSES.items():
        comp_orig = ComponentClass(pars=DEFAULT_PARS[name])
        tiny_age = 1e-10
        comp_orig.update_attribute(attributes={'age':tiny_age})

        comp_from_attr = ComponentClass(
                attributes={'mean':comp_orig.get_mean(),
                            'covmatrix':comp_orig.get_covmatrix()}
        )
        assert np.allclose(comp_orig.get_pars(), comp_from_attr.get_pars())
        assert np.allclose(comp_orig.get_covmatrix(),
                           comp_from_attr.get_covmatrix())

