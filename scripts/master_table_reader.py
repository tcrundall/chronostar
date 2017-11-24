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
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile',  dest = 'i',
                    default='data/Astrometry_with_RVs_250pc_100kms.fits',
                    help='The file being inspected')
args = parser.parse_args()
infile = args.i
stars = pyfits.getdata(infile,1)

# Display some interesting stuff 
# print(stars.columns)
print("Total number of stars: {}\n".format(stars.shape[0]))

# Display group assignment
not_groups = stars.field('Notional Group')
ngd = defaultdict(int)
for word in not_groups:
    ngd[word] += 1
print("Notional Groups from Literature")
print(ngd)
