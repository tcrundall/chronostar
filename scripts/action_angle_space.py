"""
Rough attempts to find quick and nasty way to exploit action angle
orbit calculations.

Relevant links:
    - https://galpy.readthedocs.io/en/v1.4.0/actionAngle.html#actionanglestaeckel
    - https://galpy.readthedocs.io/en/v1.4.0/orbit.html#fastchar
    - https://galpy.readthedocs.io/en/v1.4.0/actionAngle.html#accessing-action-angle-coordinates-for-orbit-instances
"""

import numpy as np

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

from galpy.actionAngle import actionAngleTorus
from galpy.potential import MWPotential2014
import sys
sys.path.insert(0, '..')
import chronostar.traceorbit as torb

xyzuvw_start = [0.,0.,25.,0.,0.,0.]
print('xyzuvw start: {}'.format(xyzuvw_start))

galpy_coords = torb.convert_cart2galpycoords(xyzuvw_start)
print('galpy start: {}'.format(galpy_coords))

aAT= actionAngleTorus(pot=MWPotential2014)

o = Orbit(vxvv=galpy_coords, ro=8., vo=220.)
o.e(analytic=True, type='staeckel', pot=MWPotential2014)

aAS = actionAngleStaeckel(pot=mp, delta=0.4)


# Om= aAT.Freqs(jr,lz,jz)


