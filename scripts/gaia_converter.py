from __future__ import print_function, division

"""
Converts all gaia stars into XYZUVW coordinates by appending new columns
to the table
"""

from astropy.table import Table

import sys
sys.path.insert(0, '..')

import chronostar.datatool as dt
import chronostar.converter as cv

gaia_table_filename = '../data/gaia_full_6d_table.fits'
gaia_table = Table.read(gaia_table_filename)

units = {'ra':'deg', 'ra_error':'mas', 'dec':'deg', 'dec_error':'mas',
         'parallax':'mas', 'parallax_error':'mas', 'pmra':'mas/yr',
         'pmra_error':'mas/yr', 'pmdec':'mas/yr', 'pmdec_error':'mas/yr',
         'radial_velocity':'km/s', 'radial_velocity_error':'km/s',
         }

# ensure columns of units
for colname in gaia_table.colnames:
    if colname in units.keys():
        gaia_table[colname].unit = units[colname]

# make room for cartesian values
dt.appendCartColsToTable(gaia_table)

# convert astrometry to cartesian
nrows = len(gaia_table)
for row_ix, row in enumerate(gaia_table):
    cv.convertRowToCartesian(row, row_ix, nrows)

gaia_table.write('../data/gaia_cartesian_full_6d_table.fits')
