#! /usr/bin/env python
"""
    Use this script to quickly examine the contents of a traceback
    pkl or fits file.
"""

import pickle
import astropy.io.fits as pyfits
from astropy.table import Table
import sys
import pdb
import numpy as np
import pandas

sys.path.insert(0,'..')

infile =\
    "../data/rave_active_star_candidates_with_TGAS_kinematics.dat"

#mydata is an array
RA, DEC, EWirt, eEWirt, log10_Age, GAIA_SOURCEID, pmRA_TGAS, pmRA_error_TGAS, pmDE_TGAS, pmDE_error_TGAS, parallax_TGAS, parallax_error_TGAS, HRV, eHRV = np.loadtxt(infile, comments="#", unpack=True) 

t = Table(
    [
    GAIA_SOURCEID,          #names
    RA, #RAdeg
    DEC, #DEdeg
    np.array(parallax_TGAS) * 1000 ,#Plx [mas], orig .dat file in as
    np.array(parallax_error_TGAS) * 1000, #e_Plx [mas], orig .dat file in as
    HRV,   #RV [km/s]
    eHRV,
    pmRA_TGAS,
    pmRA_error_TGAS,
    pmDE_TGAS,
    pmDE_error_TGAS,
    EWirt,          #activity
    eEWirt,         #error of activity
    log10_Age,      #approx age
    ],
    names=('Name', 'RAdeg','DEdeg','Plx','e_Plx','RV','e_RV',
           'pmRA','e_pmRA','pmDE','e_pmDE', 'EWirt', 'eEWirt', 'log10_Age')
    )

# Display information about the stars
new_file = infile[:-3] + "pkl"
with open(new_file, 'w') as fp:
    pickle.dump(t, fp)
