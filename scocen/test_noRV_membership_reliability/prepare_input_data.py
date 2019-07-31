"""
Prepare known Sco-Cen members with known radial velocities.
Make two sets: One with radial velocities and one with very large radial velocity errors.
"""

import numpy as np
from astropy.table import Table, vstack, join

datafile = '../data_table_cartesian_100k.fits' # SHOULD BE CARTESIAN

data_original = Table.read(datafile)
#print data_original

data_original['radial_velocity'] = 0.0
data_original['radial_velocity_error'] = 1e+5

data_original.write('data_table_cartesian_100k_with_all_big_rv_errors.fits', overwrite=True, format='fits')

# Select only Tim's members
def prepare_Tims_data():
    memb_usco = np.load('../usco_res/final_membership.npy')
    data_usco = Table.read('../usco_res/usco_run_subset.fit')
    for i in range(memb_usco.shape[1] - 1):
        data_usco['Comp_USco_%d' % (i + 1)] = memb_usco[:, i]
    data_usco['Comp_bg_USco'] = memb_usco[:, -1]
    data_usco['nonbg'] = -(data_usco['Comp_bg_USco']-1.0)

    memb_ucl = np.load('../ucl_res/final_membership.npy')
    data_ucl = Table.read('../ucl_res/ucl_run_subset.fit')
    for i in range(memb_ucl.shape[1] - 1):
        data_ucl['Comp_UCL_%d' % (i + 1)] = memb_ucl[:, i]
    data_ucl['Comp_bg_UCL'] = memb_ucl[:, -1]
    data_ucl['nonbg'] = -(data_ucl['Comp_bg_UCL'] - 1.0)

    memb_lcc = np.load('../lcc_res/final_membership.npy')
    data_lcc = Table.read('../lcc_res/lcc_run_subset.fit')
    for i in range(memb_ucl.shape[1] - 1):
        data_lcc['Comp_LCC_%d' % (i + 1)] = memb_lcc[:, i]
    data_lcc['Comp_bg_LCC'] = memb_lcc[:, -1]
    data_lcc['nonbg'] = -(data_lcc['Comp_bg_LCC'] - 1.0)

    data_memb = vstack([data_usco, data_ucl])
    data_memb = vstack([data_memb, data_lcc])

    # Find the highest probability value
    nonbg_usco = -(data_memb['Comp_bg_USco']-1.0)
    nonbg_ucl = -(data_memb['Comp_bg_UCL']-1.0)
    nonbg_lcc = -(data_memb['Comp_bg_LCC']-1.0)

    data_memb['nonbg_USco'] = nonbg_usco
    data_memb['nonbg_UCL'] = nonbg_ucl
    data_memb['nonbg_LCC'] = nonbg_lcc

    #We only need members
    mask = data_memb['nonbg']>0.5

    return data_memb[mask]

scocen_members = prepare_Tims_data()

keys=['source_id', 'background_log_overlap', 'Comp_USco_1', 'Comp_USco_2', 'Comp_USco_3', 'Comp_USco_4', 'Comp_bg_USco', 'nonbg', 'Comp_UCL_1', 'Comp_UCL_2', 'Comp_UCL_3', 'Comp_UCL_4', 'Comp_bg_UCL', 'Comp_LCC_1', 'Comp_LCC_2', 'Comp_LCC_3', 'Comp_LCC_4', 'Comp_bg_LCC', 'nonbg_USco', 'nonbg_UCL', 'nonbg_LCC']
scocen_members_essential_cols = scocen_members[keys]

data = join(data_original, scocen_members_essential_cols, keys='source_id')
print data
print data.colnames

data.write('scocen_members_with_artificially_broken_radial_velocities_for_comparison.fits', format='fits', overwrite=True)