#! /usr/bin/env python
"""
At the moment, a simple script to test usage of trace_forward.
For some number of times [nts] between the range 0 and max_age, traceforward
a star from initial position [s] to each of these times.

Two approaches are used, one is an iterative approach by calculating position
at ts[i+1] by tracing forward from ts[i] (xyzyvw)

The other approach (xyzuvw2) traces forward from the intial positon each
time

Unsure why, but the iterative approach varies quite wildly from the continuous
approach, despite being consistnet for the first few steps.
"""

import sys

sys.path.insert(0,'..') #hacky way to get access to module

from chronostar import traceback as tb
import numpy as np
import matplotlib.pyplot as plt
import pdb

#s = [9.27,-5.96,-13.59,-10.94,-16.25,-9.27]
s = [0,0,0,0,5,0]

nts = 1000        # number of times
max_age = 10000
age_step = max_age*1.0 / nts
ts = np.linspace(0,max_age-age_step,nts)
xyzuvw = np.zeros((nts,6))
xyzuvw[0]  = s
xyzuvw2 = np.zeros((nts,6))
xyzuvw2[0] = s

stepped_ages = np.zeros(nts) # array for debugging which
                             # ages are stepped through

# This loop iteratively builds a list of positions by tracing forward from 
# the previously calculated position
for i in range(nts-1):
    stepped_ages[i+1] = stepped_ages[i] + age_step
    xyzuvw[i+1] = tb.trace_forward(xyzuvw[i], age_step, solarmotion=None)

# This loop calculates the positions by tracing forward from the initial
# posiiton each time
for i, age in enumerate(ts[1:]):
    if i%50 == 0:
        print ("{} of {} done".format(i, nts))
    xyzuvw2[i+1] = tb.trace_forward(s, age, solarmotion=None)

#pdb.set_trace()
plt.clf()
plt.plot(xyzuvw2[:,0], xyzuvw2[:,1], label="cont")
plt.plot(xyzuvw[:,0], xyzuvw[:,1], label="iter")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc='best')
plt.show()

# the threshold is quite low since errors grow quite quickly
assert(np.max(xyzuvw - xyzuvw2) < 0.1)

sys.path.insert(0,'.') # reinserting home directory into path

