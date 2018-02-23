#! /usr/bin/env python
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
import numpy as np

lsr_orbit = Orbit(vxvv=[1.0,0,1,0,0.,0],vo=220,ro=8,solarmotion='schoenrich')
lsr_orbit.integrate(np.linspace(0,10,10), MWPotential2014)

true_solar_motion = np.array([-11.1, -12.24, -7.25])

U = lsr_orbit.U(0)[0]
V = lsr_orbit.V(0)[0]
W = lsr_orbit.W(0)[0]
results = np.array([U, V, W])

for i, vel in enumerate(['U', 'V', 'W']):
    print("For velocity {}:".format(vel))
    print("    expected: {}".format(true_solar_motion[i]))
    print("    received: {}".format(results[i]))
#print("Solar U motion: {}".format(U))
#print("Solar V motion: {}".format(V))
#print("Solar W motion: {}".format(W))



assert (np.allclose(true_solar_motion, results))
