#! /usr/bin/env python

from galpy.potential import MWPotential2014 as mp
from galpy.orbit import Orbit

import matplotlib.pyplot as plt
import numpy as np
import pdb

o = Orbit(vxvv=[1., 0.1, 1.1, 0., 0.1])
ts = np.linspace(0,100,10000)
o.integrate(ts, mp, method='odeint')

pdb.set_trace()
plot = o.plot()
o.plotE(normed=True)
