"""
Author: Marusa Zerjal, 2019 - 07 - 15

Take Sco-Cen components fitted to 6D data and make overlaps
(using covariance matrix) with stars missing radial velocities
in order to find more Sco-Cen candidates.


"""

import numpy as np
import sys
sys.path.insert(0, '..')
print('First')
from chronostar.component import SphereComponent
print('Second')
from chronostar import tabletool
print('Third')
from chronostar import expectmax

print('START')

# Read all components
c_usco = np.load('usco_res/final_comps.npy')
c_ucl = np.load('ucl_res/final_comps.npy')

c = np.vstack((c_usco, c_ucl))
print c
print c.shape

# Read Gaia data including both stars with known and missing radial velocities
data_table = tabletool.read('../data/ScoCen_box_result.fits')
print('DATA READ')
# Set missing radial velocities to some value, and their errors to something very big

# Set missing radial velocities (nan) to 0
data_table['radial_velocity'] = np.nan_to_num(data_table['radial_velocity'])

# Set missing radial velocity errors (nan) to 1e+10
data_table[np.isnan(data_table['radial_velocity_error'])] = 1e+10

# Convert to Cartesian
print('Convert to cartesian')
historical = 'c_XU' in data_table.colnames
# Performs conversion in place (in memory) on `data_table`
if (not 'c_XU' in data_table.colnames and not 'X_U_corr' in data_table.colnames):
    tabletool.convert_table_astro2cart(table=data_table, return_table=True)

# data_table should include background overlaps as well

print('Create data dict')
# Create data dict
data_dict = tabletool.build_data_dict_from_table(
        data_table,
        get_background_overlaps=False, # TODO change to True and provide bg ols
        historical=historical,
)

print data_dict

# Create components
#comps =



#overlaps = expectmax.get_all_lnoverlaps(data_dict, comps)

#membership_probabilities = expectmax.calc_membership_probs(overlaps)