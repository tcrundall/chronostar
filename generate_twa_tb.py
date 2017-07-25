#! /usr/bin/env python
"""Script used to generate a tb file for TWA 'core' members as proposed by
Ducourant+ 2014.

Using astrometry info from Donaldson 2016, converts into preferred units,
then applies chronostar.traceback.traceback() (?) function.
"""

import chronostar.traceback as tb
import chronostar.groupfitter as gf
import chronostar._overlap as ov
import numpy as np
import pdb
import pickle
from csv import reader
from astropy.table import Table, Column
import argparse
try:
    import astropy.io.fits as pyfits
except:
    import pyfits

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--local',  dest = 'l', action='store_true',
                    help='Set this flag if not running on raijin')
args = parser.parse_args()
local = args.l

if not local:
    data_dir    = '/short/kc5/data/'
else:
    data_dir    = 'data/'

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

infile = open(data_dir + 'Donaldson16_TWA_astrometry.csv', 'r')
data = []
counter = 0

# Go through each line, if first element in line is next core star, record
# the astrometry
for line in reader(infile):
    data += [line]
data = np.array(data)

nTWAstars = data.shape[0]
RA  = np.zeros(nTWAstars)
DEC = np.zeros(nTWAstars)

# converting ra and dec measurments to decimal
for i in range(nTWAstars):
    RA[i], DEC[i] = HMS2deg(data[i][1], data[i][2])

# Make an (astropy?) table, and store as .pkl file
Plx, e_Plx, pmDE, e_pmDE, pmRA, e_pmRA =\
    data[:,3], data[:,4], data[:,11], data[:,12], data[:,9], data[:,10]
RV, e_RV = data[:,6], data[:,7]
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
times = np.linspace(0,15,40)

# generates a traceback file which is a pickled tuple
# (star_table, times, XYZUVWs, XYZUVW_covariance_matrices)
xyzuvw = tb.traceback(t,times,savefile=data_dir+'TWA_traceback_15Myr.pkl')
