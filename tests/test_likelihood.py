"""
Check a bunch of things from the likelihood module, a set of functions that
encapsulate emcee fitting of a Component
"""

from astropy.table import Table
import astropy.units as u
import astropy.constants as const
import numpy as np

import sys
sys.path.insert(0,'..')
from chronostar import likelihood
from chronostar.component import SphereComponent, EllipComponent
from chronostar import tabletool


def astropyCalcAlpha(dx,dv,nstars):
    """
    Assuming we have identified 100% of star mass, and that average
    star mass is 1 M_sun.

    Calculated alpha is unitless
    """
    return ((dv * u.km / u.s) ** 2 * dx * u.pc /
            (const.G * nstars * const.M_sun)).decompose().value

def test_calcAlpha():
    dx = 10.
    dv = 10.
    nstars = 100.
    # Using constant for Msun from python3's astrpy
    if sys.version[0] == '2':
        rtol=1e-3
    else:
        rtol=1e-10
    assert np.isclose(astropyCalcAlpha(dx,dv,nstars),
                      likelihood.calc_alpha(dx, dv, nstars),
                      rtol=rtol)


def test_lnprior():
    dim = 6
    mean = np.zeros(dim)
    covmatrix = np.identity(dim)
    age = 10.
    sphere_comp = SphereComponent(attributes={
        'mean':mean,
        'covmatrix':covmatrix,
        'age':age,
    })
    memb_probs = np.ones(10)

    assert np.isfinite(likelihood.lnprior(sphere_comp, memb_probs))

    # Now increase age to something ridiculous
    sphere_comp.update_attribute(attributes={
        'age':1e10,
    })
    assert np.isinf(likelihood.lnprior(sphere_comp, memb_probs))

    # Try an EllipComponent with a non-symmetrical covariance matrix
    covmatrix[0,1] = 1.01
    # covmatrix[1,0] = 100
    ellip_comp = EllipComponent(attributes={
        'mean':mean,
        'covmatrix':covmatrix,
        'age':age,
    })
    assert np.isinf(likelihood.lnprior(ellip_comp, memb_probs))

    # Try an EllipComponent with a very broken correlation value
    covmatrix[0,1] = 1.01
    covmatrix[1,0] = 1.01
    ellip_comp = EllipComponent(attributes={
        'mean':mean,
        'covmatrix':covmatrix,
        'age':age,
    })
    assert np.isinf(likelihood.lnprior(ellip_comp, memb_probs))


def test_get_lnoverlaps():
    """
    Confirms that star-component overlaps get smaller as stars get further
    away
    """
    dim = 6
    mean = np.zeros(dim)
    covmatrix = np.identity(dim)
    age = 1e-10
    sphere_comp = SphereComponent(attributes={
        'mean':mean,
        'covmatrix':covmatrix,
        'age':age,
    })

    dx_offsets = [0., 1., 10.]

    star_comps = []
    for dx_offset in dx_offsets:
        star = SphereComponent(attributes={
            'mean':sphere_comp.get_mean()+np.array([dx_offset,0.,0.,0.,0.,0.]),
            'covmatrix':sphere_comp.get_covmatrix(),
            'age':sphere_comp.get_age(),
        })
        star_comps.append(star)

    nstars = len(star_comps)
    dummy_table = Table(data=np.arange(nstars).reshape(nstars,1),
                        names=['name'])
    tabletool.appendCartColsToTable(dummy_table)

    for star_comp, row in zip(star_comps, dummy_table):
        tabletool.insertDataIntoRow(row,
                                    star_comp.get_mean(),
                                    star_comp.get_covmatrix(),
                                    cartesian=True,
                                    )
    ln_overlaps = likelihood.get_lnoverlaps(sphere_comp, data=dummy_table)

    # Checks that ln_overlaps is descending
    assert np.allclose(ln_overlaps, sorted(ln_overlaps)[::-1])






