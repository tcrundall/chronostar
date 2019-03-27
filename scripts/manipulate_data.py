"""
Data manipulation.
"""

from __future__ import print_function, division, unicode_literals

from astropy.table import Table

data=Table.read('../data/no_rv_paper/beta_Pictoris_with_gaia_small_everything_final.fits')

sigma_rv = [30.0 for x in range(len(data['radial_velocity_error_best']))]
data['radial_velocity_error_best'] = sigma_rv

data.write('../data/no_rv_paper/beta_Pictoris_with_gaia_small_everything_final_big_fake_rv_errors.fits', format='fits')