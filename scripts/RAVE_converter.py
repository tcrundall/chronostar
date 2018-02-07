#! /usr/bin/env python
"""
    Use this script to quickly examine the contents of a traceback
    pkl or fits file.

    TODO: Check units of parallax
"""

import pickle
import astropy.io.fits as pyfits
from astropy.table import Table
import sys
import pdb
import numpy as np
import pandas

sys.path.insert(0,'..')


restricted_data_file =\
    "../data/rave_active_stars_distance_better_than_5_percent_3k_stars.dat"
complete_data_file =\
    "../data/rave_active_stars_distance_better_than_20_percent_16k_stars.dat"
infiles = [restricted_data_file, complete_data_file]

#mydata is an array
#RA, DEC, EWirt, eEWirt, log10_Age, GAIA_SOURCEID, pmRA_TGAS, pmRA_error_TGAS,\
# pmDE_TGAS, pmDE_error_TGAS, parallax_TGAS, parallax_error_TGAS, HRV, eHRV\
# = np.loadtxt(infile, dtype=str, comments="#", unpack=True)
for infile in infiles:
    GAIA_SOURCEID, HRV, eHRV, RA, DEC, pmRA, epmRA, pmDE, epmDE, parallax, eparallax\
        = np.loadtxt(
            infile, dtype=str, comments="#", unpack=True, delimiter=';'
        )

    float_vec = np.vectorize(float)

    t = Table(
        [
        GAIA_SOURCEID,          #names
        float_vec(RA), #RAdeg
        float_vec(DEC), #DEdeg
        float_vec(np.array(parallax))*1000 , #Plx [mas], .dat file in as IS THIS STILL THE CASE?!
        float_vec(np.array(eparallax))*1000, #e_Plx [mas], .dat file in as
        float_vec(HRV),   #RV [km/s]
        float_vec(eHRV),
        float_vec(pmRA),
        float_vec(epmRA),
        float_vec(pmDE),
        float_vec(epmDE),
    #    float_vec(EWirt),          #activity
    #    float_vec(eEWirt),         #error of activity
    #    float_vec(log10_Age),      #approx age
        ],
        names=('Name', 'RAdeg','DEdeg','Plx','e_Plx','RV','e_RV',
               'pmRA','e_pmRA','pmDE','e_pmDE',) #'EWirt', 'eEWirt', 'log10_Age')
        )
    assert type(t['Name'][0]) == np.string_

    # Display information about the stars
    new_file = infile[:-3] + "pkl"
    with open(new_file, 'w') as fp:
        pickle.dump(t, fp)
