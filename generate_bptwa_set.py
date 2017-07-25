#! /usr/bin/env python
"""Not written yet...

TO DO:
    - edit this file to generate a tb file with BPMG and TWA candidate members
    for testing EM fitting with
"""

import pickle
import argparse
import astropy.io.fits as pyfits
import sys
import pdb
import numpy as np
import chronostar.traceback as tb
from collections import defaultdict

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

# work out how to join these two tables...
pdb.set_trace()

# need to compile parallaxes from BPMG together... the ones in the 
# 'parallax_1' column appear to have smaller errors in general...


# still a work in progress:
RA     = np.append(np.array(twa_stars.columns['RAdeg']), bpmg_stars['ra_adopt'])
DEC    = np.append(np.array(twa_stars.columns['DEdeg']), bpmg_stars['dec_adopt'])
Plx    = np.append(np.array(twa_stars.columns['Plx'  ]), bpmg_stars['ra_adopt'])
e_Plx  = np.append(np.array(twa_stars.columns['e_Plx']), bpmg_stars['ra_adopt'])
RV     = np.append(np.array(twa_stars.columns['RV'   ]), bpmg_stars['ra_adopt'])
e_RV   = np.append(np.array(twa_stars.columns['e_RV' ]), bpmg_stars['ra_adopt'])
pmDE   = np.append(np.array(twa_stars.columns['pmRA' ]), bpmg_stars['ra_adopt'])
e_pmDE = np.append(np.array(twa_stars.columns['e_pmRA']), bpmg_stars['ra_adopt'])
pmRA   = np.append(np.array(twa_stars.columns['pmDE' ]), bpmg_stars['ra_adopt'])
e_pmRA = np.append(np.array(twa_stars.columns['e_pmDE']), bpmg_stars['ra_adopt'])

names=('RAdeg','DEdeg','Plx','e_Plx','RV','e_RV',
           'pmRA','e_pmRA','pmDE','e_pmDE')

times = np.linspace(0,25,40)

save_file = 'data/TWA_BPMG_traceback_25Myr.pkl'
tb.traceback(joined_table, times, savefile=save_file)
