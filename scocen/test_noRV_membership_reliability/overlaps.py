"""
Author: Marusa Zerjal, 2019 - 07 - 15

Take Sco-Cen components fitted to 6D data and make overlaps
(using covariance matrix) with stars missing radial velocities
in order to find more Sco-Cen candidates.

MZ: It fails in python2 (cannot import emcee).

"""

import numpy as np
import sys
sys.path.insert(0, '../../')
from chronostar.component import SphereComponent
from chronostar import tabletool
from chronostar import expectmax
from astropy.table import Table, vstack, join



c = np.load('../all_nonbg_scocen_comps.npy') # including LCC
print('components', c.shape)
print('Are there duplicate components?')

datafile = 'scocen_members_with_artificially_broken_radial_velocities_for_comparison_with_all_tims_members.fits'
data_table = tabletool.read(datafile)

# This table is masked. Unmask:
data_table = data_table.filled()

print('DATA READ', len(data_table))
historical = 'c_XU' in data_table.colnames

"""
# Read background overlaps
ln_bg_ols = np.loadtxt('bgols_scocen_with_artificially_broken_radial_velocities_multiprocessing.dat')
print('len bg_ols', len(ln_bg_ols), 'len data_table', len(data_table))

bg_lnol_colname = 'background_log_overlap'
print('Background overlaps: insert column')
tabletool.insert_column(data_table, ln_bg_ols, bg_lnol_colname, filename=datafile)

print('Print bg ols to cartesian table')
data_table.write('data_table_cartesian_with_bg_ols_artificially_broken_radial_velocities.fits', overwrite=True, format='fits')
"""
############################################################################
############ COMPONENT OVERLAPS ############################################
############################################################################

print('Create data dict')
# Create data dict for real
data_dict = tabletool.build_data_dict_from_table(
        data_table,
        get_background_overlaps=True,
        historical=historical,
)

# Create components
comps = [SphereComponent(pars=x) for x in c]

# COMPONENT OVERLAPS
overlaps = expectmax.get_all_lnoverlaps(data_dict, comps)
print('overlaps.shape', overlaps.shape, len(comps))

# MEMBERSHIP PROBABILITIES
membership_probabilities = np.array([expectmax.calc_membership_probs(ol) for ol in overlaps])

# Create a table
for i in range(membership_probabilities.shape[1]-1):
    data_table['comp_overlap_%d' % (i + 1)] = membership_probabilities[:, i]
data_table['comp_overlap_bg'] = membership_probabilities[:, -1]

# Print data
print('WRITE A TABLE WITH PROBABILITIES')
data_table.write('scocen_members_with_artificially_broken_radial_velocities_for_comparison_with_all_tims_members_with_component_probabilities.fits', format='fits', overwrite=True)
print(data_table)
