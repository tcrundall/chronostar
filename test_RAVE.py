"""This is a test script for tracing back beta Pictoris stars"""

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import pylab as p
import chronostar.traceback as traceback
#plt.ion()

#Read in numbers for RAVE. For HIPPARCOS, we have *no* radial velocities in general.
t=Table.read('data/ravedr4.dat', readme='data/RAVE_DR4_ReadMe.txt', format='ascii.cds')
#Brief filter to reduce the number of stars in the following calculation
t = t[(t['Dist'] < 0.25) & (t['Dist'] > 0)]

#Sorts the stars and finds the ones within 200pc of the Pleiades
vec_wd = np.vectorize(traceback.withindist)
rve = t[vec_wd((t['RAdeg']),(t['DEdeg']),(t['Dist']),0.1)]
#Calculate the pleiades model - centre of mass of stars 10pc away
pl = t[vec_wd((t['RAdeg']),(t['DEdeg']),(t['Dist']),0.01)]
centre_pl = []
col = ['RAdeg','DEdeg','plx','pmRAU4','pmDEU4','HRV']
for i in col:
    a = np.mean(pl[i])
    centre_pl = np.append(centre_pl,a)
#pleiades model (including motions) from Simbad database
pl_from_simbad = np.array([56.75,24.1167,7.34214,19.17,-44.82,3.503])

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
#Centre of mass - Black line
plt_COM = traceback.traceback2(centre_pl,times)
plt.plot(plt_COM[:,dim1],plt_COM[:,dim2],'ko')
#Simbad - Cyan line
plt_simbad = traceback.traceback2(pl_from_simbad,times)
plt.plot(plt_simbad[:,dim1],plt_simbad[:,dim2],'co')


plt.savefig('plot.png')
plt.show()
