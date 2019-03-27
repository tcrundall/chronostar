"""
Data manipulation.
"""

from __future__ import print_function, division, unicode_literals

from astropy.table import Table

data=Table.read('../data/no_rv_paper/beta_Pictoris_with_gaia_small_everything_final.fits')

print(data.colnames)