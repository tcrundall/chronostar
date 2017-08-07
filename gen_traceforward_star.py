#! /usr/bin/env python
from chronostar import traceback as tb
import numpy as np
import matplotlib.pyplot as plt
import pdb

s = [9.27,-5.96,-13.59,-10.94,-16.25,-9.27]

nts = 10
max_age = 2
age_step = max_age*1.0 / nts
ts = np.linspace(0.01,max_age,nts)
xyzuvw = np.zeros((nts,6))
xyzuvw[0] = s

xyzuvw2 = np.zeros((nts,6))

# for some reason, doing it recursively seems to double add
# the  [0,0,25,11.1,12.24,7.25] (xyzuvw_sun) to the xyzuvw of the star...
for i in range(nts-1):
    pdb.set_trace()
    xyzuvw[i+1] = tb.trace_forward(xyzuvw[i], age_step)

for i, age in enumerate(ts):
    xyzuvw2[i] = tb.trace_forward(s, age)
#pdb.set_trace()
plt.plot(xyzuvw[:,0], xyzuvw[:,1])
plt.plot(xyzuvw2[:,0], xyzuvw2[:,1])
plt.show()
