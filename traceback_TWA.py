"""Generates a tb file for TWA members by first building an astropy table 
from Donaldson+ 2016 measured astrometry.
"""
#! /usr/bin/env python

import chronostar.traceback as tb
import chronostar.groupfitter as gf
import chronostar._overlap as ov
import numpy as np
import pdb
import pickle
from csv import reader
from astropy.table import Table, Column
try:
    import astropy.io.fits as pyfits
except:
    import pyfits

# Nasty hacky way of checking if on Raijin
onRaijin = True
try:
    dummy = None
    pickle.dump(dummy, open("/short/kc5/dummy.pkl",'w'))
except:
    onRaijin = False

if onRaijin:
    location = "/short/kc5/"
else:
    location = "data/"

# Taken from www.bdnyc.org/2012/10/decimal-deg-to-hms/
def HMS2deg(ra='', dec=''):
    RA, DEC, rs, ds = '', '', 1, 1
    if dec:
        D, M, S = [float(i) for i in dec.split()]
        if str(D)[0] == '-':
            ds, D = -1, abs(D)
        deg = D + (M/60) + (S/3600)
        DEC = '{0}'.format(deg*ds)

    if ra:
        H, M, S = [float(i) for i in ra.split()]
        if str(H)[0] == '-':
            rs, H = -1, abs(H)
        deg = (H*15) + (M/4) + (S/240)
        RA = '{0}'.format(deg*rs)

    if ra and dec:
        return (RA, DEC)
    else:
        return RA or DEC

infile = open(location + 'Donaldson16_TWA_astrometry.csv', 'r')
data = []
for line in reader(infile):
    data += [line]
data = np.array(data)

nTWAstars = data.shape[0]
RA  = np.zeros(nTWAstars)
DEC = np.zeros(nTWAstars)

# converting ra and dec measurments to decimal
for i in range(nTWAstars):
    RA[i], DEC[i] = HMS2deg(data[i][1], data[i][2])

Plx, e_Plx, pmDE, e_pmDE, pmRA, e_pmRA =\
    data[:,3], data[:,4], data[:,11], data[:,12], data[:,9], data[:,10]
RV, e_RV = data[:,6], data[:,7]

# making an astropy table
t = Table(
    [data[:,0],
     RA.astype(np.float),    
     DEC.astype(np.float),
     Plx.astype(np.float),
     e_Plx.astype(np.float),
     RV.astype(np.float),
     e_RV.astype(np.float),
     pmRA.astype(np.float),
     e_pmRA.astype(np.float),
     pmDE.astype(np.float),
     e_pmDE.astype(np.float)],
    names=('Name', 'RAdeg','DEdeg','Plx','e_Plx','RV','e_RV',
           'pmRA','e_pmRA','pmDE','e_pmDE')
    )
times = np.linspace(0,20,40)
tb.traceback(t,times,plotit=True,savefile=location + 'TWA_traceback_20Myr.pkl')
