"""This is a test script for tracing back beta Pictoris stars"""

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from chronostar.error_ellipse import plot_cov_ellipse
import chronostar.traceback as traceback
plt.ion()

#Read in numbers for beta Pic. For HIPPARCOS, we have *no* radial velocities in general.
bp=Table.read('data/betaPic.csv')
#Remove bad stars. "bp" stands for Beta Pictoris.
bp = bp[np.where([ (n.find('6070')<0) & (n.find('12545')<0) & (n.find('Tel')<0) for n in bp['Name']])[0]]
times = np.linspace(0,20,21)

#Which dimensions do we plot? 0=X, 1=Y, 2=Z
dims = [0,1]
dim1=dims[0]
dim2=dims[1]
xoffset = np.zeros(len(bp))
yoffset = np.zeros(len(bp))

#Some hardwired plotting options.
if (dims[0]==0) & (dims[1]==1):
    yoffset[0:10] = [6,-8,-6,2,0,-4,0,0,0,-4]
    yoffset[10:] = [0,-8,0,0,6,-6,0,0,0]
    xoffset[10:] = [0,-4,0,0,-15,-10,0,0,-20]
    axis_range = [-70,60,-40,120]

if (dims[0]==1) & (dims[1]==2):
    axis_range = [-40,120,-30,100]
    text_ix = [0,1,4,7]
    xoffset[7]=-15

#Trace back orbits with plotting enabled.
tb = traceback.TraceBack(bp)
tb.traceback(times,xoffset=xoffset, yoffset=yoffset, axis_range=axis_range, dims=dims,plotit=True,savefile="results/traceback_save.pkl")


#Error ellipse for the association. This comes from "fit_group.py".
xyz_cov = np.array([[  34.25840977,   35.33697325,   56.24666544],
       [  35.33697325,   46.18069795,   66.76389275],
       [  56.24666544,   66.76389275,  109.98883853]])
xyz = [ -6.221, 63.288, 23.408]
cov_ix1 = [[dims[0],dims[1]],[dims[0],dims[1]]]
cov_ix2 = [[dims[0],dims[0]],[dims[1],dims[1]]]
plot_cov_ellipse(xyz_cov[cov_ix1,cov_ix2],[xyz[dim1],xyz[dim2]],alpha=0.5,color='k')

plt.savefig('plot.png')
plt.show()