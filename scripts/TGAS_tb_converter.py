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

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile',  dest = 'i',
                    help='The file being inspected')
args = parser.parse_args()

infile = "data/TGAS_traceback_165Myr_small.fits"

stars = pyfits.getdata(infile,1)
times = pyfits.getdata(infile,2)
xyzuvw = pyfits.getdata(infile,3)
xyzuvw_cov = pyfits.getdata(infile,4)

# Display information about the traceback ages
print("Data:\n  Ages:\n    Start:\t{}\n    End:\t{}\n    Count:\t{}\n"
        .format(times[0], times[-1], times.shape[0]))

# Display information about the stars
print("Stars:\n  Count:\t{}\n"
        .format(xyzuvw_cov.shape[0]))
new_infile = infile[:-4] + "pkl"
pickle.dump((stars,times,xyzuvw,xyzuvw_cov), open(new_infile, 'w'))
