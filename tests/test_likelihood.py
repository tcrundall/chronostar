"""
Check a bunch of things from the likelihood module, a set of functions that
encapsulate emcee fitting of a Component
"""

import astropy.units as u
import astropy.constants as const
import numpy as np

import sys
sys.path.insert(0,'..')
from chronostar import likelihood
from chronostar.component import Component


def astropyCalcAlpha(dx,dv,nstars):
    """
    Assuming we have identified 100% of star mass, and that average
    star mass is 1 M_sun.

    Calculated alpha is unitless

    TODO: Astropy slows things down a very much lot. Remove it!
    """
    return ((dv * u.km / u.s) ** 2 * dx * u.pc /
            (const.G * nstars * const.M_sun)).decompose().value

def test_calcAlpha():
    dx = 10.
    dv = 10.
    nstars = 100.
    assert np.isclose(astropyCalcAlpha(dx,dv,nstars),
                      likelihood.calcAlpha(dx,dv,nstars))

