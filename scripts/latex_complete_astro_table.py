from __future__ import print_function, division

"""
Generates a basic latex table of stellar astrometry
from master_table and membership array
"""


import numpy as np
from astropy.table import Table


save_file_name = '../results/tables/final_bpmg_members_complete_astro_table.tex'

# rdir = '../results/em_fit/beta_Pictoris_with_gaia_small_1.0/3/A/final/'
# z_file = rdir + 'final_membership.npy'

table_file = '../data/beta_Pictoris_with_gaia_small_everything_final.fits'
beta_table = Table.read(table_file)

# sub_table = star_pars['table'][np.where(np.isfinite(star_pars['table']['comp_A']))]
# sub_table.sort('comp_A')

# bpmg_comp_ix = 0
# bpmg_memb_probs = np.load(z_file)[:,bpmg_comp_ix]

# threshold = 0.5
# bpmg_table_mask = np.array(star_pars['indices'])[np.where(bpmg_memb_probs > threshold)]

# columns: Gaia DR2 id, main designation, ra, dec, parallax, pmra, pmdec, rv,
#          membership prob, references
# main_designation
col_names = [
    'main_designation',
    'ra',
    # 'ra_error',
    'dec',
    # 'dec_error',
    'parallax',
    # 'parallax_error',
    'pmra',
    # 'pmra_error',
    'pmdec',
    # 'pmdec_error',
    'radial_velocity_best',
    # 'radial_velocity_error_best',
    'comp_A',
    'radial_velocity_ref',
    'lit_membership_ref',
]

titles = {
    'main_designation':['Main', 'Designation'],
    'ra':['R.A.', '[deg]'],
    'dec':['Decl', '[deg]'],
    'parallax':['Parallax', '[mas]'],
    'pmra':['$\\mu_\\alpha \\cos \\delta$', '[mas$\un{yr}{-1}$]'],
    'pmdec':['$\\mu_\\delta$', '[mas$\un{yr}{-1}$]'],
    'radial_velocity_best':['RV', '[$\kms$]'],
    'comp_A':['Comp. A', 'Memb. Prob.'],
    'radial_velocity_ref':['RV ref', ''],
    'lit_membership_ref':['Prev.','$\\beta$PMG ref'],
}

astro_w_errs_cols = ['parallax', 'pmra', 'pmdec', 'radial_velocity_best']
def getErrorKey(astro_colname):
    assert np.isin(astro_colname, astro_w_errs_cols)
    if astro_colname != 'radial_velocity_best':
        return astro_colname + '_error'
    else:
        return 'radial_velocity_error_best'

def convertRow(row):
    """
    Helper function that converts a row into a string for writing to latex file
    """
    out_str = ''
    out_str += row['main_designation'] + ' & '
    out_str += '${:.3f}$ & '.format(row['ra'])
    out_str += '${:.3f}$ & '.format(row['dec'])
    for col_name in astro_w_errs_cols:
        col_err_name = getErrorKey(col_name)
        error = row[col_err_name]
        value = row[col_name]

        if np.isnan(error):
            out_str += ' &'
        else:
            # get digit place of error's most significant figure
            prec = -int(np.floor(np.log10(abs(error))))
            if prec >= 0:
                val_place = '{:.' + str(prec) + 'f}'
            else:
                # TODO: work out how to round numbers sensibly...
                val_place = '{:.' + str(prec) + 'f}' # <-- this line is wrong atm
                # val_place = '{:' + str(abs(prec)) + '.}'
            out_str += ('$ ' + val_place + '\pm' + val_place + '$ &').format(value, error)
    if np.isnan(row['comp_A']):
        out_str += ' &'
    else:
        out_str += ' {} &'.format(row['comp_A'])
    out_str += ' {} &'.format(row['radial_velocity_ref'])
    out_str += ' {} '.format(row['lit_membership_ref'])
    out_str += ' \\\\\n'
    return out_str


with open(save_file_name, 'w') as fp:
    fp.write('\\begin{tabular}{l|' + 6*'r|' + 3*'l|' + '}\n')
    fp.write('\\hline\n')
    header_fst_str = ''
    for col_name in col_names:
        if np.isin(col_name, titles.keys()):
            print(col_name)
            header_fst_str += ' {} &'.format(titles[col_name][0])
    # remove trailing '&'
    fp.write(header_fst_str[:-1] + ' \\\\\n')

    header_snd_str = ''
    for col_name in col_names:
        if np.isin(col_name, titles.keys()):
            print(col_name)
            header_snd_str += ' {} &'.format(titles[col_name][1])
    # remove trailing '&'
    fp.write(header_snd_str[:-1] + ' \\\\\n')
    fp.write('\\hline\n')


    for row in beta_table[:12]:
        if row['companion'] == 'False' and np.isnan(row['comp_A']):
            pass
        else:
            fp.write(convertRow(row))

    fp.write('\\hline\n')
    fp.write('\\end{tabular}\n')

