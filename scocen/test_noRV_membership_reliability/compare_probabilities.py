"""
Prepare known Sco-Cen members with known radial velocities.
Make two sets: One with radial velocities and one with very large radial velocity errors.
"""

import numpy as np
from astropy.table import Table, vstack, join

data = Table.read('../data_table_cartesian_with_bg_ols_and_component_overlaps.fits')
datanorv = Table.read('scocen_members_with_artificially_broken_radial_velocities_for_comparison.fits')

print data.colnames
print datanorv.colnames

#keys=