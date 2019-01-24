from __future__ import print_function, division

"""
Checking all the new BPMG members from Gaia DR2 stars to see if they're
featured as members elsewhere, also just to get their main designations.
"""

from astropy.table import Table
import csv
import numpy as np
import sys
sys.path.insert(0, '..')
import chronostar.synthesiser as syn
import chronostar.datatool as dt
import chronostar.coordinate as coord

import banyan_parser as bp

def readShkolnik():
    """
    Read Shkolnik into an array

    File header:
    2MASS J, Other, Ref., SpT, Ref., EW[Halpha], Ref., EW[Li], Ref., RV, Ref.,
        Grav., Ref., pi, Ref., RV, Halpha, Li, BPMG,

    Note:
        2MASS ids have no preceding J. They are simply [8digits]-/+[7digits]
    """
    shkol_file = '../data/shkolnik_all_poss_bpmg.txt'

    # with open(shkol_file, 'r') as csvfile:
    fp = open(shkol_file, 'r')
    lit_memb_range = (9,155)
    shkol_memb_range = (158,199)
    shkol_raw = fp.readlines()
    fp.close()

    shkol_clean =\
        [row.strip().split('\t') for row in shkol_raw[slice(*lit_memb_range)]] \
        + [row.strip().split('\t') for row in shkol_raw[slice(*shkol_memb_range)]] \

    # ensure each row has same length (by appending 'cdots' to final two
    # columns as required
    max_row_length = max(len(row) for row in shkol_clean)
    for row in shkol_clean:
        len_diff = max_row_length - len(row)
        row += len_diff * ['cdots']

    return np.array(shkol_clean)


def writeGaiaIds(table):
    """

    :param table:
    :return:
    """
    output = '../data/beta_Pictoris_with_gaia_small_gaia_ids.txt'
    fp = open(output, 'w')
    for sid in table['source_id']:
        if sid != 'nan':
            fp.write('Gaia DR2 {}\n'.format(sid))
    fp.close()


def readSimIds():
    """
    For this funciton to work, must first call `writeGaiaIds` on table
    in question, then submit resulting file to SIMBAD (with options
    include 2mass identifier) then save file to required filename
    """
    filename = '../data/beta_Pictoris_with_gaia_small_sim_ids.txt'

    fp = open(filename, 'r')
    sim_ids_raw = fp.readlines()
    fp.close()
    # name_break_ix = 57
    sim_data_range = (7,602)
    # sim_data_raw = [row for row in sim_ids_raw[slice(*sim_data_range)]]
    two_mass_range = (40,56)
    gaia_id_range = (13,32)
    other_name_range = (58,84)
    primary_name_range = (33,56)

    star_labels = {}
    for row in sim_ids_raw[slice(*sim_data_range)]:
        gaia_id = int(row[slice(*gaia_id_range)])
        primary_name = row[slice(*primary_name_range)].strip()
        if primary_name[:6] == '2MASS ':
            two_mass_id = row[slice(*two_mass_range)]
            primary_name = None
        else:
            two_mass_id = None
        other_name = row[slice(*other_name_range)].strip()

        star_labels[gaia_id] = (two_mass_id, primary_name or other_name)

    return star_labels


rdir = '../results/em_fit/beta_Pictoris_wgs_inv2_5B_res/'
final_memb_file = rdir + 'final_membership.npy'
# bp_star_pars_file = '../data/beta_Pictoris_with_gaia_small_xyzuvw.fits'
bp_star_pars_file = '../data/bpmg_w_nearby_gaia_memberships_magnitudes.fits'
gagne_star_pars_file = '../data/gagne_bonafide_full_kinematics_with_lit_and' \
                       '_best_radial_velocity_comb_binars.fits'

z = np.load(final_memb_file)
bp_sp = dt.loadDictFromTable(bp_star_pars_file)
gg_sp = dt.loadDictFromTable(gagne_star_pars_file)

banyan_data = bp.readBanyanTxtFile()

banyan_upper_ix = 38
new_chronostar_membs = (np.where(z[banyan_upper_ix:,0]>0.5)[0] + banyan_upper_ix,)

# -- insert ra dec for stars not matched with Gaia --

# first incorporate BANYAN radec into all members table
bd_ras = []
bd_des = []
assert len(banyan_data) == len(gg_sp['table'])
for row in banyan_data:
    if row[3] == '':
        bd_ras.append(None)
        bd_des.append(None)
    else:
        bd_ras.append(coord.convertRAtoDeg(row[3], row[4], row[5]))
        bd_des.append(coord.convertDEtoDeg(row[6], row[7], row[8]))

bd_ras_arr = np.array(bd_ras, dtype=np.float64)
bd_des_arr = np.array(bd_des, dtype=np.float64)

gg_sp['table']['gagne_ra'] = bd_ras_arr
gg_sp['table']['gagne_ra'].unit = 'deg'
gg_sp['table']['gagne_de'] = bd_des_arr
gg_sp['table']['gagne_de'].unit = 'deg'

for row in bp_sp['table']:
    if np.isnan(row['ra']):
        gg_ix = np.where(gg_sp['table']['Main designation'] ==\
                         row['Main designation'])[0][0]
        row['ra'] = gg_sp['table'][gg_ix]['gagne_ra']
        row['dec'] = gg_sp['table'][gg_ix]['gagne_de']

# INCORPORTATE 2MASS IDS AND MAIN IDENTIFIERS
import pdb;
# pdb.set_trace()
star_labels = readSimIds()
for row in bp_sp['table']:
    if row['Main designation'] == 'N/A' and row['source_id'] != 'nan':
        gaia_dr2_id = np.int64(row['source_id'])
        try:
            row['Main designation'] = star_labels[gaia_dr2_id][1]
            two_mass_id = star_labels[gaia_dr2_id][0]
            if two_mass_id:
                two_mass_id = 'J' + two_mass_id
            row['2MASS'] = two_mass_id
            row['Gaia DR2'] = gaia_dr2_id
        except KeyError:
            pass
            # print('Gaia DR2 {} not on Simbad'.format(gaia_dr2_id))


# Manually insert 2MASS id from ra-dec search
#   (There are <=5 stars with Gaia DR2 ids that don't have 2MASS ids from
#    SIMBAD, most probably simply because the stars haven't been connected yet)
# row_ix, RA, DEC:
# (220, 304.42903844677187, -21.109887755962337) --> 2MASS J20174317-2106346 (looked at)
# (233, 271.0674361851081, -30.30804794215021) --> 2MASS J18041617-3018280 (looked at)
#   (found through VIZIER, not on SIMBAD, 0.9 arcsec difference in DEC, which
#    almost perfectly consistent with proper motion in dec)
# (238, 314.0116539175749, -17.181880624834275) --> 2MASS J20560274-1710538 (looked at)
# (284, 297.0689547751161, -27.342434650132887) --> 2MASS J19481651-2720319 (looked at)
#   (found through VIZIER, not on SIMBAD, proper motions accoutn for 1 arcesc
#    discrepency in position
# (295, 269.6512792271908, -40.99132808464182) --> 2MASS J17583642-4059270 (looked at)
# (342, 229.13713500626673, -48.5434062333674) --> 2MASS J15163296-4832354 (looked at)
#   (found through VIZIER, not on SIMBAD, proper motions accoutn for 1 arcesc
#    discrepency in position
bp_sp['table'][220]['2MASS'] = 'J20174317-2106346'
bp_sp['table'][233]['2MASS'] = 'J18041617-3018280'
bp_sp['table'][238]['2MASS'] = 'J20560274-1710538'
bp_sp['table'][284]['2MASS'] = 'J19481651-2720319'
bp_sp['table'][295]['2MASS'] = 'J17583642-4059270'
bp_sp['table'][342]['2MASS'] = 'J15163296-4832354'

# in 15 years, 60 mas / year


# APPLY BPMG MEMBERSHIP STATUS
short_empty_column = np.array(len(bp_sp['table']) * [2 * ' '])
long_empty_column = np.array(len(bp_sp['table']) * [18 * ' '])

bp_sp['table']['lit_memb'] = short_empty_column
bp_sp['table']['lit_memb_ref'] = long_empty_column
# pdb.set_trace()

shkol_arr = readShkolnik()

# use the 2MASS id (in column '2MASS') where available, to get Shkolnik
# BPMG membership
overlap = 0
for i, row in enumerate(bp_sp['table']):
    # Apparently some entries have whitespace...
    # Also need to drop the initial 'J'
    two_mass_id = row['2MASS'].strip()[1:]
    if two_mass_id in shkol_arr[:,0]:
        # print("Handling {}".format(two_mass_id))
        if two_mass_id =='15553295-6010404':
            # import pdb; pdb.set_trace()
            pass
        shkol_row = shkol_arr[np.where(shkol_arr[:,0] == two_mass_id)][0]
        # print(shkol_row[-1])
        row['lit_memb'] = shkol_row[-1]
        row['lit_memb_ref'] = shkol_row[2]
        overlap += 1


print("{} stars in common with Shkolnik".format(overlap))

# Include BANYAN membership flag
for row in bp_sp['table']:
    if row['Moving group'] == 'beta Pictoris':
        row['lit_memb'] = 'Y'
        row['lit_memb_ref'] = r'BANYAN $\Sigma$'

# Gaia DR2 ids for stars allocated to BPMG by most recent BANYAN paper (XII)
new_banyan_membs = np.array([
    3292922293081928192,
    3181961503752885248,
    3290081910949989888,  # slightly brighter
    3290081906654767616,  # companion of ^^
    3209947441933983744,
    2477815222028038272,
    4093006560668685568,  # proper motion needed to confirm
    3393207610483520896,
    5924485966955008896,  # THIS STAR WE ALSO DISCOVERED
    6631762764424312960, # HD 173167, BPMG member
], dtype=np.str)

for row in bp_sp['table']:
    if row['source_id'] in new_banyan_membs:
        row['lit_memb'] = 'Y'
        prev_ref = row['lit_memb_ref'].strip()
        row['lit_memb_ref'] = 'new BY, ' + prev_ref


# Incorporate chronostar memberhsip probabilities
for i, memb_col in enumerate(z.T):
    empty_flt_column = np.array(len(bp_sp['table']) * [np.nan])
    if i == len(z.T) - 1:
        col_name = 'comp_background'
    else:
        col_name = 'comp_{}'.format(chr(ord('A') + i))
    bp_sp['table'][col_name] = empty_flt_column
    bp_sp['table'][col_name][bp_sp['indices']] = memb_col

# Incorporate other BANYAN memberships
for i, row in enumerate(bp_sp['table']):
    if row['source_id'] in gg_sp['table']['source_id']:
        row['Moving group'] =\
            gg_sp['table']['Moving group'][np.where(gg_sp['table']['source_id'] == row['source_id'])[0][0]]

# Incorporate photometrically consistent flag
#   (code copy pasted from 'scripts/read_bpmg_resuts.py'

# first, put absolute magnitude into table
bp_sp['table']['abs_g_mag'] = bp_sp['table']['phot_g_mean_mag']\
          + 5*(np.log10(1e-3*bp_sp['table']['parallax'])+1)

xs = (1.5, 3)
ys = (7.5,10.)
m = (ys[1] - ys[0]) / (xs[1] - xs[0])
c = ys[0] - m * xs[0]

def line_eq(x):
    return m*x + c

THRESHOLD = 0.2
bpmg_rows = bp_sp['table'][np.where(bp_sp['table']['Comp 0 prob'] > THRESHOLD)]
bpmg_ixs = np.where(bp_sp['table']['comp_A'] > THRESHOLD)
# abs_mag = bpmg_rows['phot_g_mean_mag']\
#           + 5*(np.log10(1e-3*bpmg_rows['parallax'])+1)


# find Chronostar membeers which are photometerically inconsistent
main_seq_stars = np.where((bp_sp['table']['comp_A'] > THRESHOLD) & (line_eq(bp_sp['table']['bp_rp']) < bp_sp['table']['abs_g_mag']))

bp_sp['table']['phot_consist'] = short_empty_column
bp_sp['table']['phot_consist'][bpmg_ixs] = 'Y'
bp_sp['table']['phot_consist'][main_seq_stars] = 'N'


# Triple check (by 2MASS identifier) that no 'new' members featured in the
# BANYAN list

new_and_consist_membs_mask =\
    np.where((bp_sp['table']['comp_A'] > THRESHOLD)
             & (np.array([el.strip() for el in bp_sp['table']['lit_memb']])=='')
             & (bp_sp['table']['phot_consist'] == 'Y'))


missing = np.where((bp_sp['table']['comp_A'] > THRESHOLD)
         & (np.array([el.strip() for el in bp_sp['table']['lit_memb']])=='')
         & (bp_sp['table']['phot_consist'] == 'Y')
         & (bp_sp['table']['2MASS'] == 'N/A'))

# for row in bp_sp['table'][new_and_consist_membs_mask]:
#     if row['2MASS'] in gg_sp['table']['2MASS']:
#         print(row)

Table.write(bp_sp['table'],
            '../data/beta_Pictoris_with_gaia_small_everything.fits',
            overwrite=True)

comp_total = len(np.where(bp_sp['table']['comp_A'] > 0.5)[0])
print()
print('Total component stars: {}'.format(comp_total))

banyan_total = len(np.where((bp_sp['table']['comp_A'] > 0.5)
                 & (bp_sp['table']['Moving group'] == 'beta Pictoris'))[0])
print('Confirmed BANYAN stars: {}'.format(banyan_total))

print()
print('Follow up BANYAN members:')
follow_total = 0
for row in bp_sp['table']:
    if row['lit_memb_ref'].split(',')[0] == 'new BY':
        print(row['2MASS'])
        follow_total += 1
print(follow_total)

print()
print('stars listed in Shkolnik')
shkol_total = 0
for row in bp_sp['table']:
    try:
        int(row['lit_memb_ref'])
        if row['comp_A'] > 0.5:
            print(row['lit_memb_ref'], row['2MASS'])
            shkol_total += 1
    except ValueError:
        pass
print(shkol_total)

print()
print('Photometrically inconsistnet')
phot_total = 0
for row in bp_sp['table']:
    if (row['comp_A'] > 0.5) and (row['lit_memb'].strip() == '') and \
            (row['phot_consist'] == 'N'):
        print(row['phot_consist'])
        phot_total += 1
print(phot_total)

print('----- summary -----')
print('Total:  {}'.format(comp_total))
print('BANYAN: {}'.format(banyan_total))
print('Follow: {}'.format(follow_total))
print('Shkol:  {}'.format(shkol_total))
print('Phot:   {}'.format(phot_total))

remaining = comp_total - banyan_total - follow_total - shkol_total - phot_total
print()
print('Remaining: {}'.format(remaining))

# TODO: maybe go through and count systematically the unique stars, for comparison

