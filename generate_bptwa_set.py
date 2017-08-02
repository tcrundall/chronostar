#! /usr/bin/env python
"""
Generates a combined traceback file of TWA and BPMG stars.
The first 38 stars are TWA stars with astormetry from Donaldson 2016.
The next 31 stars are BPMG stars with astrometry from ~Gaia.
"""

import pickle
import argparse
import astropy.io.fits as pyfits
import sys
import pdb
import numpy as np
import chronostar.traceback as tb
from collections import defaultdict
from astropy.table import Table, Column

infile = "data/Astrometry_with_RVs_250pc_100kms.fits"
master_stars = pyfits.getdata(infile,1)

# Display some interesting stuff 
# print(stars.columns)

bpmg_ixs  = np.where(master_stars['Notional Group'] == 'bPMG')
pbpmg_ixs = np.where(master_stars['Notional Group'] == 'Possible bPMG') 

all_ixs = (np.sort(np.append(bpmg_ixs[0], pbpmg_ixs[0])),)

# Read in traceback file, select out only bpmg stars, resave as a pkl
tb_file = 'data/TGAS_traceback_165Myr_small.fits'

bpmg_stars      = pyfits.getdata(tb_file,1)[all_ixs]
# bpmg_stars      = master_stars[all_ixs] # ! use this if want to retain
                                          # ! the extra information

# read in table for twa
twa_stars = pickle.load(open("data/TWA_traceback_15Myr.pkl", 'r'))[0]

RA     = np.append(np.array(twa_stars.columns['RAdeg']),
                            bpmg_stars['ra_adopt'])
DEC    = np.append(np.array(twa_stars.columns['DEdeg']),
                            bpmg_stars['dec_adopt'])
Plx    = np.append(np.array(twa_stars.columns['Plx']),
                            bpmg_stars['parallax_1'])
e_Plx  = np.append(np.array(twa_stars.columns['e_Plx']),
                            bpmg_stars['parallax_error'])
RV     = np.append(np.array(twa_stars.columns['RV']),
                            bpmg_stars['rv_adopt'])
e_RV   = np.append(np.array(twa_stars.columns['e_RV']),
                            bpmg_stars['rv_adopt_error'])
pmDE   = np.append(np.array(twa_stars.columns['pmRA']),
                            bpmg_stars['pmra_1'])
e_pmDE = np.append(np.array(twa_stars.columns['e_pmRA']),
                            bpmg_stars['pmra_error'])
pmRA   = np.append(np.array(twa_stars.columns['pmDE' ]),
                            bpmg_stars['pmdec'])
e_pmRA = np.append(np.array(twa_stars.columns['e_pmDE']),
                            bpmg_stars['pmdec_error'])

nstars = len(RA)
ntwastars = len(twa_stars)
# go through and check for BPMG stars with NaN values. Replace with alternate
for i in range(nstars):
    if Plx[i]!=Plx[i]:
        bpmg_ix = i-ntwastars
        Plx[i]    = bpmg_stars['Plx']   [bpmg_ix]
        e_Plx[i]  = bpmg_stars['e_Plx'] [bpmg_ix]
        pmDE[i]   = bpmg_stars['pmDE']  [bpmg_ix]
        e_pmDE[i] = bpmg_stars['e_pmDE'][bpmg_ix]
        pmRA[i]   = bpmg_stars['pmRA_2']  [bpmg_ix]
        e_pmRA[i] = bpmg_stars['e_pmRA'][bpmg_ix]

t = Table(
    [
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
    names=('RAdeg','DEdeg','Plx','e_Plx','RV','e_RV',
           'pmRA','e_pmRA','pmDE','e_pmDE')
    )

pickle.dump(t, open("data/TWA_BPMG_joined_astrometry.pkl", 'w'))
times = np.linspace(0,25,40)
save_file = 'data/TWA_BPMG_traceback_25Myr.pkl'
tb.traceback(t, times, savefile=save_file)
