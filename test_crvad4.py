"""This is a test script for tracing back beta Pictoris stars"""

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from chronostar.error_ellipse import plot_cov_ellipse
import chronostar.traceback as traceback
import chronostar.play as play
plt.ion()

#This needs to be fixed! Vaguely correct below.
pl = play.crvad2[play.get_pl_loc2()]
pl['HIP'].name = 'Name'
#Which times are we plotting?
times = np.linspace(0,5,11)
xoffset = np.zeros(len(pl))
yoffset = np.zeros(len(pl))
axis_range=[-500,500,-500,500]

#Which dimensions do we plot? 0=X, 1=Y, 2=Z
dims = [0,1]
dim1=dims[0]
dim2=dims[1]


#Trace back orbits with plotting enabled.
tb = traceback.TraceBack(pl)
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