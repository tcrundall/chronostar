"""This is a test script for tracing back beta Pictoris stars"""

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import pylab as p
import chronostar.traceback as traceback
#plt.ion()

#Read in numbers for RAVE. For HIPPARCOS, we have *no* radial velocities in general.
t=Table.read('crvad2.dat', readme='crvad2.ReadMe', format='ascii.cds')
t['Dist'] = 1/t['Plx']
t['RAdeg'] = 15.0*t['RAhour']
t['e_RAdeg'] = 15*t['e_RAhour']
#Brief filter to reduce the number of stars in the following calculation
t = t[(t['Dist'] < 0.25) & (t['Dist'] > 0)]
#Sorts the stars and finds the ones within 200pc of the Pleiades
vec_wd = np.vectorize(traceback.withindist)
stars = t[vec_wd((t['RAdeg']),(t['DEdeg']),(t['Dist']),0.02)]

#Calculate the pleiades model - centre of mass of stars 10pc away
pl = t[vec_wd((t['RAdeg']),(t['DEdeg']),(t['Dist']),0.02)]
#pl = pl[pl['Vmag'] < 7.0]
centre_pl = []
col = ['RAdeg','DEdeg','Plx','pmRA','pmDE','RV']
for i in col:
    x = pl[i]
    x_bar = np.mean(x)
    #sd = np.std(x)
    pl = pl[np.absolute(x - x_bar)/pl['e_%s' %i] < 5]
    print (len(x))
for i in col:
    x = pl[i]
    x_bar = np.mean(x)
    centre_pl = np.append(centre_pl,x_bar)
    
#pleiades model (including motions) from Simbad database
pl_from_simbad = np.array([56.75,24.1167,7.34214,19.17,-44.82,3.503]) # V changed from -ve to +ve

#The vector of times.
times = np.linspace(0,20,21)

#Some hardwired plotting options.
xoffset = np.zeros(len(stars))
yoffset = np.zeros(len(stars))
 
#Which dimensions do we plot? 0=X, 1=Y, 2=Z
dims = [0,1]
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
tb = traceback.TraceBack(stars)
tb.traceback(times,xoffset=xoffset, max_plot_error=150, yoffset=yoffset, axis_range=axis_range, dims=dims,plotit=True,savefile="results/traceback_save.pkl")

#Centre of mass - Black line
plt_COM = traceback.traceback2(centre_pl,times)
plt.plot(plt_COM[:,dim1],plt_COM[:,dim2],'co')
#Simbad - Cyan line
#plt_simbad = traceback.traceback2(pl_from_simbad,times)
#plt.plot(plt_simbad[:,dim1],plt_simbad[:,dim2],'co')


plt.savefig('plot.png')
plt.show()
