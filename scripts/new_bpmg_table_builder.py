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

def readShkrolnik():
    """
    Read Shkrolnik into an array
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

    return shkol_clean


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
            print('Gaia DR2 {} not on Simbad'.format(gaia_dr2_id))

# APPLY BPMG MEMBERSHIP STATUS


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

