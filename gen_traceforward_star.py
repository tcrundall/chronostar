#! /usr/bin/env python
from chronostar import traceback as tb
import numpy as np
import matplotlib.pyplot as plt
import pdb

#s = [9.27,-5.96,-13.59,-10.94,-16.25,-9.27]
s = [0,0,0,0,0,0]

nts = 10
max_age = 10
age_step = max_age*1.0 / nts
ts = np.linspace(1,max_age,nts)
xyzuvw = np.zeros((nts,6))
xyzuvw[0]  = s
xyzuvw2 = np.zeros((nts,6))
xyzuvw2[0] = s

pdb.set_trace()

# BUG!!! For whatever reason, the solar motion is being added to the end
# result twice...

# This loop iteratively builds a list of positions by tracing forward from 
# the previously calculated position
for i in range(nts-1):
    #pdb.set_trace()
    xyzuvw[i+1] = tb.trace_forward(xyzuvw[i], age_step)

# This loop calculates the positions by tracing forward from the initial
# posiiton each time
for i, age in enumerate(ts[1:]):
    xyzuvw2[i+1] = tb.trace_forward(s, age)

#pdb.set_trace()
plt.plot(xyzuvw[:,0], xyzuvw[:,1])
plt.plot(xyzuvw2[:,0], xyzuvw2[:,1])
plt.show()
