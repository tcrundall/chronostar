#! /usr/bin/env python
"""
    Use this script to quickly examine the contents of a traceback
    pkl or fits file.
"""

import pickle
import argparse
import astropy.io.fits as pyfits
import sys
import pdb
import numpy as np
import chronostar.traceback as tb
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile',  dest = 'i',
                    default='data/Astrometry_with_RVs_250pc_100kms.fits',
                    help='The file being inspected')
args = parser.parse_args()
infile = args.i
master_stars = pyfits.getdata(infile,1)

# Display some interesting stuff 
# print(stars.columns)

bpmg_ixs  = np.where(master_stars['Notional Group'] == 'bPMG')
pbpmg_ixs = np.where(master_stars['Notional Group'] == 'Possible bPMG') 

all_ixs = (np.sort(np.append(bpmg_ixs[0], pbpmg_ixs[0])),)

# Read in traceback file, select out only bpmg stars, resave as a pkl
tb_file = 'data/TGAS_traceback_165Myr_small.fits'
# overrun stars with reduced set
# stars      = pyfits.getdata(tb_file,1)[all_ixs]
stars      = master_stars[all_ixs]
times      = pyfits.getdata(tb_file,2)
xyzuvw     = pyfits.getdata(tb_file,3)[all_ixs]
xyzuvw_cov = pyfits.getdata(tb_file,4)[all_ixs]

save_file = 'data/BPMG_traceback_165Myr.pkl'
with open(save_file, 'w') as fp:
    pickle.dump((stars,times,xyzuvw,xyzuvw_cov),fp)

