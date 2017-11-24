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
                    default='data/TWA_core_traceback_15Myr.pkl',
                    help='The file being inspected')
args = parser.parse_args()

infile = args.i

if infile[-3:] == 'pkl':
    stars, times, xyzuvw, xyzuvw_cov = pickle.load(open(infile, 'r'))
elif infile[-3:] == 'fit' or infile[-4:] == 'fits':
    stars = pyfits.getdata(infile,1)
    times = pyfits.getdata(infile,2)
    xyzuvw = pyfits.getdata(infile,3)
    xyzuvw_cov = pyfits.getdata(infile,4)
else:
    print("File format {} not suitable".format(infile[-3:]))
    sys.exit()

# Display information about the traceback ages
print("Data:\n  Ages:\n    Start:\t{}\n    End:\t{}\n    Count:\t{}\n"
        .format(times[0], times[-1], times.shape[0]))

# Display information about the stars
print("Stars:\n  Count:\t{}\n"
        .format(xyzuvw_cov.shape[0]))
