"""
Turns out the table used was based on the wrong results.
This script uses the correct results from beta_P../6/E and
updates the table accordingly
"""
from astropy.table import Table
import numpy as np

import sys
sys.path.insert(0, '..')
from chronostar import tabletool

orig_table_path = '../data/paper1/beta_Pictoris_with_gaia_small_everything_final.fits'

orig_table = Table.read(orig_table_path)

res_dir = '../results/beta_Pictoris_with_gaia_small_inv2/6/E/final/'
final_memb = np.load(res_dir + 'final_membership.npy')

recons_star_pars, table_ixs =\
    tabletool.build_data_dict_from_table(orig_table, return_table_ixs=True)

# --------------------------------------------------
# --  Insert/replace membership probabilities ------
# --------------------------------------------------

# need to add new column for comp_F
# simpler just to remove all membership probability columns and append
# to end, this keeps them together without reshuffling of columns
existing_colnames = ['comp_'+char for char in 'ABCDE'] + ['comp_background']
print(existing_colnames)

for colname in existing_colnames:
    del orig_table[colname]

new_colnames = ['comp_'+char for char in 'ABCDEF'] + ['comp_background']
for ix, colname in enumerate(new_colnames):
    print(ix, colname)
    orig_table[colname] = np.nan
    orig_table[colname][table_ixs] = np.round(final_memb[:,ix], decimals=5)

# --------------------------------------------------
# --  Insert BANYAN membership allocations  --------
# --------------------------------------------------
# A little clumsy:
# - find stars that share gaia id across banyan table and bpmg_plus_nearby table
# - if a star is missing a gaia id in bpmg_plus_nearby table then it means it
#     was originally from the banyan table with bpmg membership, so set it
banyan_table = Table.read('../data/paper1/banyan_with_gaia_near_bpmg_xyzuvw.fits')

banyan_membership = np.array(len(orig_table) * [''])
banyan_membership[table_ixs] = 'N/A'
banyan_membership = np.array(banyan_membership, dtype='U22')
# banyan_membership.dtype = 'U22'

# blank_str_col = np.array(banyan_membership, dtype='U22')
for i in range(len(orig_table)):
    master_table_ix = np.where(banyan_table['source_id'] == orig_table['gaia_dr2'][i])
    if len(master_table_ix[0]) > 1:
        print(ix, master_table_ix)
    if len(master_table_ix[0]) > 0:
        banyan_membership[i] = banyan_table['Moving group'][master_table_ix[0][0]]

banyan_membership[np.where(banyan_membership == '')] = 'beta Pictoris'
orig_table['banyan_assoc'] = banyan_membership

# Update previous membership references
new_ref_ixs = {
    4050178830427649024 : '29',
    6754492966739292928 : '29',
    5963633872326630272 : '29',
    6731752867256886144 : '30',
    6723183789033085824 : '31',
}

for ix, ref in new_ref_ixs.items():
    print(orig_table['gaia_dr2'][np.where(orig_table['gaia_dr2'] == str(ix))])
    row_ix = np.where(orig_table['gaia_dr2'] == str(ix))[0]
    orig_table['lit_membership_ref'][row_ix] = ref

    print(orig_table['phot_consist'][row_ix])


save_filename = '../data/paper1/beta_Pictoris_corrected_everything.fits'

# Table.write(orig_table, save_filename)
# from astropy.io import ascii
# ascii.write(orig_table, save_filename)