"""This is a test script for tracing back beta Pictoris stars"""

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import chronostar.traceback as traceback
plt.ion()

#Read in numbers for RAVE. For HIPPARCOS, we have *no* radial velocities in general.
t=Table.read('data/ravedr4.dat', readme='data/RAVE_DR4_ReadMe.txt', format='ascii.cds')
#Brief filter to reduce the number of stars in the following calculation
t = t[(t['Dist'] < 0.35) & (t['Dist'] > 0)]
#Sorts the stars and finds the ones within 200pc of the Pleiades
vec_wd = np.vectorize(traceback.withindist)
rve = t[vec_wd((t['RAdeg']),(t['DEdeg']),(t['Dist']),0.1)] 

#Calculate the pleiades model
pl = t[vec_wd((t['RAdeg']),(t['DEdeg']),(t['Dist']),0.01)]
pl2 = []
col = ['RAdeg','DEdeg','plx','pmRAU4','pmDEU4','HRV']
for i in col:
    a = np.mean(pl[i])
    pl2 = np.append(pl2,a)

#The vector of times.
times = np.linspace(0,20,21)

#Some hardwired plotting options.
xoffset = np.zeros(len(rve))
yoffset = np.zeros(len(rve))
 
#Which dimensions do we plot? 0=X, 1=Y, 2=Z
dims = [1,2]
dim1=dims[0]
dim2=dims[1]

if (dims[0]==0) & (dims[1]==1):
#    yoffset[0:10] = [6,-8,-6,2,0,-4,0,0,0,-4]
#    yoffset[10:] = [0,-8,0,0,6,-6,0,0,0]
#    xoffset[10:] = [0,-4,0,0,-15,-10,0,0,-20]
    axis_range = [-500,500,-500,500]

if (dims[0]==1) & (dims[1]==2):
    axis_range = [-500,500,-500,500]
    text_ix = [0,1,4,7]
#    xoffset[1]=-15 

#Trace back orbits with plotting enabled.
tb = traceback.TraceBack(rve)
tb.traceback(times,xoffset=xoffset, yoffset=yoffset, axis_range=axis_range, dims=dims,plotit=True,savefile="results/traceback_save.pkl")
pl_ln = traceback.traceback2(pl2,times)
plt.plot(pl_ln[:,dim1],pl_ln[:,dim2],'co')

plt.savefig('plot.png')
plt.show()