"""
There is a bug in current version of Galpy.
This test is here simply to demonstrate it.
"""

import galpy
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
import numpy as np

def test_correctInitialSolarMotion():
    age = 10 # Myr
    SCHOENRICH_SOLAR_MOTION = np.array([-11.1, -12.24, -7.25])
    ntimesteps = 10
    lsr_orbit = Orbit(vxvv=[1.0,0,1,0,0.,0],vo=220,ro=8,solarmotion='schoenrich')
    lsr_orbit.integrate(np.linspace(0,age,ntimesteps), MWPotential2014)

    U = lsr_orbit.U(0)
    V = lsr_orbit.V(0)
    W = lsr_orbit.W(0)

    results = np.array([U, V, W]).reshape(3)
    for i, vel in enumerate('UVW'):
        print("For velocity {}:".format(vel))
        print("    expected: {}".format(SCHOENRICH_SOLAR_MOTION[i]))
        print("    received: {}".format(results[i]))

    assert (np.allclose(SCHOENRICH_SOLAR_MOTION, results)),\
        '!!! Using galpy version {} Need galpy version 1.1 !!!'.format(
            galpy.__version__
        )
