from __future__ import print_function, division

"""
To be applied to 'beta_Pictoris... ...everything.fits', the output
to `new_bpmg_table_builder.py`. This script applies the final tidying to the
table, rewriting column names, inserting units, merging unneeded elements,
rewriting blanks etc

Changes:
- remove a bunch of cols
- rename a bunch of cols
- standardise string label for missing data ('NULL')
- rewrite membership probs lower than 1e-5 as 0.
- establish required references, then replace numbering
    - rvs and literature membership prepped
    - todo: spectral type
- reorder columns into intuitive order
"""

import numpy as np
import pandas as pd
from astropy.table import Table, hstack, join
import sys
sys.path.insert(0, '..')
import chronostar.datatool as dt

if __name__ == '__main__':
    input_file = '../data/beta_Pictoris_with_gaia_small_everything.fits'
    output_file = '../data/beta_Pictoris_with_gaia_small_everything_final.fits'
    my_banyan_table_file = '../data/gagne_bonafide_full_kinematics_with_lit_' \
                           'and_best_radial_velocity_comb_binars_with_banyan_' \
                           'radec.fits'
    banyan_file = '../data/banyan_data.txt'

    banyan_data = Table.read(banyan_file, format='ascii')
    sp = dt.loadDictFromTable(input_file)
    master_sp = dt.loadDictFromTable(my_banyan_table_file)
    master_table = master_sp['table']
    beta_table = sp['table']

    # insert new columns into master table
    new_cols = [
        'r_RVel',
        'r_SpT',
    ]
    for col_name in new_cols:
        master_table[col_name] = banyan_data[col_name]

    # use a join to get the references
    beta_table['r_RVel'] = np.array(len(beta_table) * '', dtype='S10')
    beta_table['r_SpT'] = np.array(len(beta_table) * '', dtype='S3')

    # Manually go through and cross match stars, first by 2MASS id then by Gaia
    # DR2 id. (Obviously this is doable with a table operation but for the life
    # of me I can't figure it out...)
    for row in beta_table:
        if row['Moving group'] == 'beta Pictoris':
            if np.isin(row['2MASS'].strip(), np.char.strip(master_table['2MASS'])):
                master_mask = np.where(
                    np.char.strip(master_table['2MASS']) == row['2MASS'].strip()
                )
                row['r_RVel'] = master_table[master_mask]['r_RVel'][0]
                row['r_SpT'] = master_table[master_mask]['r_SpT'][0]
            elif np.isin(row['source_id'].strip(),
                       np.char.strip(master_table['source_id'])):
                master_mask = np.where(
                    np.char.strip(master_table['source_id']) == row['source_id'].strip()
                )
                row['r_RVel'] = master_table[master_mask]['r_RVel'][0]
                row['r_SpT'] = master_table[master_mask]['r_SpT'][0]

            if row['radial_velocity_best_flag'] == 'Gaia':
                row['r_RVel'] = '999'

    if True:
        # Removing redundant columns
        redundant_cols = [
            'Gaia DR2',
            'Moving group',
            'Comp 0 prob',
            'Comp 1 prob',
            'Comp 2 prob',
            'Comp 3 prob',
            'Comp 4 prob',
            'designation',
            'radial_velocity_zwitter',
            'radial_velocity_error_zwitter',
            'radial_velocity_galah_flag',
            'radial_velocity_rave',
            'radial_velocity_error_rave',
            'tmass_oid',
            'original_ext_source_id',
            'angular_distance',
            'r_est',
            'r_lo',
            'r_hi',
            'r_len',
            'result_flag',
            'modality_flag',
        ]
        for col_name in redundant_cols:
            del(beta_table[col_name])

        # Rename columns
        column_renames = {
            'source_id' : 'gaia_dr2_id',
        }
        for orig_col_name, new_col_name in column_renames.items():
            beta_table[new_col_name] = beta_table[orig_col_name]
            del(beta_table[orig_col_name])


        # replace all missing string values with '' (empty) for consistency
        null_labels = ['NULL', 'N/A', 'none', 'None', 'nan']
        for col_name in beta_table.keys():
            # check if stored as string
            if beta_table[col_name].dtype.char == 'S':
                # strip whitespace
                beta_table[col_name] = np.char.strip(beta_table[col_name])
                # print('Replacing null elements in {}'.format(col_name))
                none_mask = np.where(np.isin(beta_table[col_name], null_labels))
                # print(len(none_mask[0]))
                beta_table[col_name][none_mask] = ''

        # rewrite memberships that are less than 1e-5 as 0
        membership_cols = [
            'comp_A',
            'comp_B',
            'comp_C',
            'comp_D',
            'comp_E',
            'comp_background',
        ]
        for col_name in membership_cols:
            beta_table[col_name] = np.round(beta_table[col_name], 5)

        # relabelling references
        # Astrometry (position, pm, parallax) all from Gaia DR2
        # RV from various (see original BANYAN \Sigma table)
        # Membership from various ('lit_memb_ref')

        # Mostly from Shkolnic reference ids
        memb_ref_dict = {
            '1':'Zuckerman et al.(2001 b)',
            '2':'Malo et al.(2013)',
            '5':'Malo et al.(2014 a)',
            '7':'Kiss et al.(2011)',
            '9':'Elliott et al.(2016)',
            '10':'Schlieder et al.(2010)',
            '12':'Moor et al.(2006)',
            '13':'Alonso - Floriano et al.(2015 a)',
            '14':'Elliott et al.(2014)',
            '16':'Torres et al.(2008)',
            '22':'Moor et al.(2013)',
            'BANYAN $\\Sigma$' : 'gagne et al. 2018a',
            'new BY' : 'gagne et al. 2018b',
            # '100':'gagne et al.(2018 a)',
            # '101':'gagne et al.(2018 b)',
        }

        chron_memb_ref = {}
        chron_memb_ix = 1
        for k,v in memb_ref_dict.items():
            chron_memb_ref[k] = (chron_memb_ix, v)
            chron_memb_ix += 1

        rv_ref_dict = {
            '8':'Faherty et al. (2016)',
            '9':'Shkolnik et al. (2012)',
            '14':'Malo et al. (2014)',
            '20':'Gontcharov (2006)',
            '23':'Anderson & Francis (2012)',
            '30':'Torres et al. (2006)',
            '34':'Kharchenko et al. (2007)',
            '36':'Montes et al. (2001)',
            '49':'Valenti & Fischer (2005)',
            '51':'Song et al. (2003)',
            '59':'Torres et al. (2009)',
            '60':'Kiss et al. (2011)',
            '63':'Allers et al. (2016)',
            '64':'Shkolnik et al. (2017)',
            '999':'gaia DR2',
        }

        chron_rv_ref = {}
        for k,v in rv_ref_dict.items():
            chron_rv_ref[k] = (chron_memb_ix, v)
            chron_memb_ix += 1

        # Save text file of table citations
        ref_filename = 'chron_ref_list.txt'
        with open(ref_filename, 'w') as ref_file:
            ref_file.write('BPMG membership references\n')
            for val in sorted(chron_memb_ref.values()):
                ref_file.write('({}) {},\n'.format(val[0], val[1]))
            ref_file.write('RV references\n')
            for val in sorted(chron_rv_ref.values()):
                ref_file.write('({}) {},\n'.format(val[0], val[1]))

        ref_list = chron_memb_ref.values() + chron_rv_ref.values()

        def convertRVref(rv_ref):
            """
            Reads in a string from 'r_RVel' col
            that is either int, '-' or '', parses it
            and returns Chronostar's equivalent reference.

            Make sure
            """
            if rv_ref == '-' or rv_ref == '':
                return ''
            else:
                return str(chron_rv_ref[rv_ref][0])

        def convertSTref(st_ref):
            """
            Reads in a string from 'r_SpT' col that is either int, '-' or '',
            parses it and returns Chronostar's equivalent reference

            TODO: build Spectral Type reference dict
            """
            chron_st_ref = {}
            if st_ref == '-' or st_ref == '':
                return ''
            else:
                return str(chron_st_ref[st_ref][0])

        def convertMembRef(memb_ref):
            """
            Reads in a string from 'r_RVel' col
            that is either int, '-' or '', parses it
            and returns Chronostar's equivalent reference.

            Make sure
            """
            output_str = ''
            if memb_ref == '-' or memb_ref == '':
                return output_str
            else:
                for el in memb_ref.split(','):
                    output_str += '{},'.format(chron_memb_ref[el.strip()][0])
                if output_str[-1] == ',':
                    output_str = output_str[:-1]
                return output_str


        new_ref_cols = [
            'radial_velocity_ref',
            'lit_membership_ref',
        ]

        nbeta_stars = len(beta_table)
        blank_ref_column = np.array(nbeta_stars*[''], dtype='S5')
        for col_name in new_ref_cols:
            beta_table[col_name] = blank_ref_column

        for row in beta_table:
            if np.isnan(row['radial_velocity_best']):
                row['radial_velocity_best'] = row['radial_velocity']
                row['radial_velocity_error_best'] = row['radial_velocity_error']
                row['r_RVel'] = '999'
            row['radial_velocity_ref'] = convertRVref(row['r_RVel'])
            row['lit_membership_ref'] = convertMembRef(row['lit_memb_ref'])


    final_col_renames = {
        'Main designation':'main_designation',
        'gaia_dr2_id':'gaia_dr2',
        'Other':'other',
        'Companion':'companion',
        'Spectral type':'spectral_type',
    }
    for orig_col_name, new_col_name in final_col_renames.items():
        beta_table[new_col_name] = beta_table[orig_col_name]

    # beta table column names
    new_order = [
        #'Main designation',
        'main_designation',
        '2MASS',
        #'gaia_dr2_id',
        'gaia_dr2',
        'AllWISE',
        #'Other',
        'other',
        #'Companion',
        'companion',
        'ra',
        'ra_error',
        'dec',
        'dec_error',
        'parallax',
        'parallax_error',
        'pmra',
        'pmra_error',
        'pmdec',
        'pmdec_error',
        'ra_dec_corr',
        'ra_parallax_corr',
        'ra_pmra_corr',
        'ra_pmdec_corr',
        'dec_parallax_corr',
        'dec_pmra_corr',
        'dec_pmdec_corr',
        'parallax_pmra_corr',
        'parallax_pmdec_corr',
        'pmra_pmdec_corr',
        'parallax_over_error',
        'ref_epoch',
        # 'radial_velocity',# remove
        # 'radial_velocity_error',# remove
        'radial_velocity_best',
        'radial_velocity_error_best',
        # 'radial_velocity_best_flag',# remove
        # 'radial_velocity_lit', # remove
        # 'radial_velocity_error_lit', # remove
        'l',
        'b',
        'ecl_lon',
        'ecl_lat',
        'X',
        'Y',
        'Z',
        'U',
        'V',
        'W',
        'dX',
        'dY',
        'dZ',
        'dU',
        'dV',
        'dW',
        'c_XY',
        'c_XZ',
        'c_XU',
        'c_XV',
        'c_XW',
        'c_YZ',
        'c_YU',
        'c_YV',
        'c_YW',
        'c_ZU',
        'c_ZV',
        'c_ZW',
        'c_UV',
        'c_UW',
        'c_VW',
        #'Spectral type',
        'spectral_type',
        'approx_mass',
        'astrometric_primary_flag',
        'teff_val',
        'phot_bp_mean_mag',
        'phot_bp_mean_flux',
        'phot_rp_mean_mag',
        'phot_g_mean_mag',
        'abs_g_mag',
        'bp_rp',
        # 'lit_memb',
        # 'lit_memb_ref',
        'comp_A',
        'comp_B',
        'comp_C',
        'comp_D',
        'comp_E',
        'comp_background',
        'phot_consist',
        # 'r_RVel',
        # 'r_SpT',
        'radial_velocity_ref',
        'lit_membership_ref'
    ]

    beta_table = beta_table[new_order]

    # confirm no duplicates
    # (found one, hardcoded the removal)
    # dup_2mass = 'J20174317-2106346'
    beta_false_2mass_match_ix = 220
    beta_table['2MASS'][beta_false_2mass_match_ix] = ''

    mys = pd.Series(beta_table['2MASS'][np.where(beta_table['2MASS'] != '')])
    assert(len(mys[mys.duplicated()]) == 0)

    beta_table.write(output_file, overwrite=True)


memb_left_overs = {
    '3':'Lepine & Simon(2009)',
    '4':'Gagne et al.(2015 a)',
    '6':'Gagne et al.(2015 b)',
    '8':'Binks & Jeffries(2016)',
    '11':'Song et al.(2003)',
    '15':'Rodriguez et al.(2014)',
    '17':'Riedel et al.(2014)',
    '18':'Schlieder et al.(2012 a)',
    '19':'Teixeira et al.(2009)',
    '20':'Schlieder et al.(2012 b)',
    '21':'Malo et al.(2014 b)',
    '23':'Liu et al.(2016)',
    '24':'Barrado y Navascues et al.(1999)',
    '25':'Liu et al.(2013)',
    '26':'AL13)',
    '27':'Binks & Jeffries(2014)',
    '28':'Gray et al.(2003)',
    '29':'Lopez - Santiago et al.(2010)',
    '30':'Zuckerman & Song(2004)',
    '31':'Torres et al.(2006)',
    '32':'Kordopatis et al.(2013)',
    '33':'Alcala et al.(2000)',
    '34':'Riaz et al.(2006)',
    '35':'Houk & Smith - Moore(1988)',
    '36':'Shkolnik et al.(2009)',
    '37':'Teixeira et al.(2009)',
    '38':'Stephenson(1986)',
    '39':'Hawley et al.(1997)',
    '40':'Hipparcos Catalog',
    '41':'Reid et al.(2008)',
    '42':'Cruz et al.(2009)',
    '43':'Shkolnik et al.(2012)',
    '44':'Lepine et al.(2013)',
    '45':'Reid et al.(2007)',
    '46':'Alcala et al.(1996)',
    '47':'White et al.(2007)',
    '48':'Song et al.(2012)',
    '49':'Herbig & Bell(1988)',
    '50':'Martin et al.(2010)',
    '51':'Gaidos et al.(2014)',
    '52':'Reid et al.(2002)',
    '53':'da Silva et al.(2009)',
    '54':'Kraus et al.(2014)',
    '55':'Faherty et al.(2016)',
    '56':'Montes et al.(2001)',
    '57':'Bailey et al.(2012)',
    '58':'Macintosh et al.(2015)',
    '59':'Gizis et al.(2002)',
    '60':'Gontcharov(2006)',
    '61':'Strassmeier & Rice(2000)',
    '62':'Wilson(1953)',
    '63':'Zuckerman & Webb(2000)',
    '64':'Allers et al.(2016)',
    '65':'Gaia Collaboration et al.(2016)',
    '66':'Faherty et al.(2013)',
    '67':'van Leeuwen(2007)',
    '68':'Schneider et al.(2017)',
    '100':'Gagne et al. (2018 a)',
    '101':'Gagne et al. (2018 b)',
}


rv_left_overs = {
    '1':'Gagne et al. (2015b)',
    '2':'Liu et al. (2016)',
    '3':'Reiners & Basri (2009)',
    '4':'Jaschek et al. (1964)',
    '5':'Gaia Collaboration et al. (2016a)',
    '6':'Bobylev et al. (2006)',
    '7':'Zuckerman & Song (2004)',
    '10':'Lepine et al. (2013)',
    '11':'Malo et al. (2013)',
    '12':'White et al. (2007)',
    '13':'Monet et al. (2003)',
    '15':'Schlieder et al. (2010)',
    '16':'van Leeuwen (2007)',
    '17':'Ivanov (2008)',
    '18':'Tokovinin & Smekhov (2002)',
    '19':'Garrison & Gray (1994)',
    '21':'Bystrov et al. (1994)',
    '22':'Egret et al. (1992)',
    '24':'Houk & Cowley (1975)',
    '25':'Zuckerman et al. (2011)',
    '26':'Blake et al. (2010)',
    '27':'Chubak & Marcy (2011)',
    '28':'Kordopatis et al. (2013)',
    '29':'Hog et al. (2000)',
    '31':'Zacharias et al. (2004a)',
    '32':'Gray et al. (2006)',
    '33':'Martin & Brandner (1995)',
    '35':'Messina et al. (2010)',
    '37':'Reid et al. (2004)',
    '38':'Kunder et al. (2017)',
    '39':'Terrien et al. (2015)',
    '40':'Knapp et al. (2004)',
    '41':'Dupuy & Liu (2012)',
    '42':'Gagne et al. (2015a)',
    '43':'Dieterich et al. (2014)',
    '44':'Roser et al. (2008)',
    '45':'Abt & Morrell (1995)',
    '46':'Zuckerman et al. (2001a)',
    '47':'Zacharias et al. (2010)',
    '48':'Riedel et al. (2014)',
    '50':'Holmberg et al. (2007)',
    '52':'Macintosh et al. (2015)',
    '53':'Zacharias et al. (2004b)',
    '54':'Allers & Liu (2013)',
    '55':'Lepine & Simon (2009)',
    '56':'Bobylev & Bajkova (2007)',
    '57':'Gizis et al. (2002)',
    '58':'Bonnefoy et al. (2013)',
    '61':'Corbally (1984)',
    '62':'Liu et al. (2013)',
    '65':'Gray et al. (2003)',
    '66':'Eggl et al. (2013)',
    '67':'King et al. (2003)',
    '68':'Levato & Abt (1978)',
}

