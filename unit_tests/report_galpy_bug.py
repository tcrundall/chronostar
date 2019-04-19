"""
There is a bug in current version of Galpy.
This test is here simply to demonstrate it.
"""

import galpy
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
import numpy as np

print('Using galpy version {}'.format(galpy.__version__))

# Purely for reference: the expected solar motion as provided by Schoenrich
REFERENCE_SCHOENRICH_SOLAR_MOTION = np.array([-11.1, -12.24, -7.25])

# Set the age to be a full galactic revolution
age = -2*np.pi # galpy time units
ntimesteps = 10
times = np.linspace(0.,age,ntimesteps)

# vxvv is in cylindrical coordinates here. [R,vR,vT(,z,vz,phi)]
lsr_orbit = Orbit(vxvv=[1.0001,0,1,0,0,0],vo=220,ro=8,solarmotion='schoenrich')
lsr_orbit.integrate(times, MWPotential2014)

# Manually extract the initial velocities
U = lsr_orbit.U(0)
V = lsr_orbit.V(0)
W = lsr_orbit.W(0)

results = np.array([U, V, W]).reshape(3)
print('-- Inspecting the initialised velocities --')
for i, vel in enumerate('UVW'):
    print("For velocity {}:".format(vel))
    print("    expected: {}".format(REFERENCE_SCHOENRICH_SOLAR_MOTION[i]))
    print("    received: {}".format(results[i]))

print("Since we initialised at the LSR we expect z to stay constant")
print("z: {}".format([lsr_orbit.z(t) for t in np.linspace(0,age,10)]))
print("Since z is constant, we expect W to be constant")
print("z: {}".format([lsr_orbit.W(t) for t in np.linspace(0,age,10)]))

import sys
sys.path.insert(0, '..')
from chronostar import traceorbit

xyzuvw = traceorbit.convert_galpycoords2cart(lsr_orbit.getOrbit(), times)


# assert (np.allclose(REFERENCE_SCHOENRICH_SOLAR_MOTION, results)),\
#    '!!! Using galpy version {} Need galpy version 1.1 !!!'.format(
#        galpy.__version__
#    )
