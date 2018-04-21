#! /usr/bin/env python
"""Script used to generate a dedicated tb file for BPMG candidate members.
Simply uses the master astrometry file which still includes the 'Notional
Group' column, identifies the indices of all stars listed as '(Possible) bPMG',
and extracts the tb data corresponding to those indices.

Currently only set up to run on misfit
"""

import pickle
import argparse
import astropy.io.fits as pyfits
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile',  dest = 'i',
                    default='data/Astrometry_with_RVs_250pc_100kms.fits',
                    help = 'Input traceback file')
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

# init_pars = (stars, times, xyzuvw, xyzuvw_cov)
# with open("data/BPMG_initial_XYZUVW.pkl", 'w') as fp:
#     pickle.dump(init_pars, fp)

save_file = 'data/BPMG_traceback_165Myr.pkl'
with open(save_file, 'w') as fp:
    pickle.dump((stars,times,xyzuvw,xyzuvw_cov),fp)
