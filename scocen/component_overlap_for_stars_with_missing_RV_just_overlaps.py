"""
Author: Marusa Zerjal, 2019 - 07 - 15

Take Sco-Cen components fitted to 6D data and make overlaps
(using covariance matrix) with stars missing radial velocities
in order to find more Sco-Cen candidates.

MZ: It fails in python2 (cannot import emcee).

"""

import numpy as np
import sys
sys.path.insert(0, '..')
from chronostar.component import SphereComponent
from chronostar import tabletool
from chronostar import expectmax
from astropy.table import Table, vstack, join



c = np.load('all_nonbg_scocen_comps.npy') # including LCC
print('components', c.shape)
print('Are there duplicate components?')

# Read Gaia data including both stars with known and missing radial velocities
datafile = 'data_table_cartesian_100k.fits'
data_table = tabletool.read(datafile)

#TODO DELETE THIS
# Just because I don't have bg_ols for stars with missing RV yet!!!
mask = data_table['radial_velocity_error']<5000
data_table=data_table[mask]

print('DATA READ', len(data_table))
historical = 'c_XU' in data_table.colnames

ln_bg_ols = np.loadtxt('ln_bg_ols.dat')

bg_lnol_colname = 'background_log_overlap'
print('Background overlaps: insert column')
tabletool.insert_column(data_table, ln_bg_ols, bg_lnol_colname, filename=datafile)

print('Print bg ols to cartesian table')
data_table.write('data_table_cartesian_with_bg_ols_tmp.fits', overwrite=True, format='fits')


# TODO: DELETE THIS
data_table = data_table[:4]

############################################################################
############################################################################


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
print(comps)


# COMPONENT OVERLAPS
overlaps = expectmax.get_all_lnoverlaps(data_dict, comps)
#print(overlaps)
#print('overlaps.shape', overlaps.shape)

membership_probabilities = np.array([expectmax.calc_membership_probs(ol) for ol in overlaps])
#print(membership_probabilities)
#print(membership_probabilities.shape)

for i in range(membership_probabilities.shape[1]):
    data_table['comp_overlap_%d' % (i + 1)] = membership_probabilities[:, i]
#for i in range(4):
#    data_table['comp_overlap_usco%d'%(i+1)]=membership_probabilities[:,i]
#for i in range(4):
#    data_table['comp_overlap_ucl%d'%(i+1)]=membership_probabilities[:,i+4]

print(data_table)



######################################################################################
######################################################################################
####  Compare with membership probabilities of stars with known radial velocities ####
######################################################################################
######################################################################################

memb_usco = np.load('usco_res/final_membership.npy')
data_usco = Table.read('usco_res/usco_run_subset.fit')
for i in range(memb_usco.shape[1]-1):
    data_usco['Comp_USco_%d'%(i+1)] = memb_usco[:,i]
data_usco['Comp_bg'] = memb_usco[:,-1]

memb_ucl = np.load('ucl_res/final_membership.npy')
data_ucl = Table.read('ucl_res/ucl_run_subset.fit')
for i in range(memb_ucl.shape[1]-1):
    data_ucl['Comp_UCL_%d'%(i+1)] = memb_ucl[:,i]
data_ucl['Comp_bg'] = memb_ucl[:,-1]

data_memb = vstack([data_usco, data_ucl])
print(data_memb)

print(type(data_table['source_id'][0]))
print(type(data_memb['source_id'][0]))

d = join(data_table, data_memb, keys='source_id')
print(d)