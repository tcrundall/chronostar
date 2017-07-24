""" Generates a tb file for BPMG stars."""

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
#from chronostar.error_ellipse import plot_cov_ellipse
import chronostar.traceback as traceback
import pickle
plt.ion()

times = np.linspace(0,2,3)
nts = len(times)
print (times)
bp = Table.read('data/betaPic.csv')
bp = bp[np.where([ (n.find('6070')<0) & (n.find('12545')<0) & (n.find('Tel')<0) for n in bp['Name']])[0]]
bp=bp[1:3]
params = np.array([bp['RAdeg'],bp['DEdeg'],bp['Plx'],bp['pmRA'],bp['pmDE'],bp['RV']])
p = np.transpose(params)

tb = traceback.TraceBack(bp)
tb.traceback(times,savefile="trial_save.pkl")
stars,times,xyzuvwb,xyzuvw_cov = pickle.load(open('trial_save.pkl'))
xyzuvwf = [traceback.traceforward((xyzuvwb[0])[nts-1],times)]
for i in np.arange(1,len(bp)):
    newxyz = [traceback.traceforward((xyzuvwb[i])[nts-1],times)]
    #x = np.flipud(x)
    xyzuvwf = np.concatenate((xyzuvwf,newxyz),axis=0)
print(xyzuvwb)
print (xyzuvwf)
print(p) 
