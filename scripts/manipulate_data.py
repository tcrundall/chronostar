"""
Data manipulation.
"""

from __future__ import print_function, division, unicode_literals

from astropy.table import Table
import numpy as np

data=Table.read('../data/no_rv_paper/bpmg_run_subset.fits')

data['radial_velocity_error'] = [100.0 for x in range(len(data))]

data['radial_velocity'] = 42.0 * np.random.randn(len(data), 1) -2.8

print(data['radial_velocity'])

data.write('../data/no_rv_paper/bpmg_run_subset_fake_rv_errors.fits', format='fits')