from __future__ import print_function, division

"""
Generates a basic latex table of stellar astrometry
from master_table and membership array
"""

import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.retired2.datatool as dt

save_file_name = '../results/tables/bpmg_members_astro_table.tex'

rdir = '../results/em_fit/beta_Pictoris_with_gaia_small_1.0/3/A/final/'
z_file = rdir + 'final_membership.npy'

table_file = '../data/beta_Pictoris_with_gaia_small_xyzuvw.fits'
star_pars = dt.loadDictFromTable(table_file)

bpmg_comp_ix = 0
bpmg_memb_probs = np.load(z_file)[:,bpmg_comp_ix]

threshold = 0.05
bpmg_table_mask = np.array(star_pars['indices'])[np.where(bpmg_memb_probs > threshold)]

# columns: Gaia DR2 id, main designation, ra, dec, parallax, pmra, pmdec, rv,
#          membership prob, references
col_names = [
    'source_id',
    'Main designation',
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
    # 'radial_velocity',
    # 'radial_velocity_error',
    'radial_velocity_best',
    'radial_velocity_error_best',
    'radial_velocity_best_flag',
    # 'X',
    # 'Y',
    # 'Z',
    # 'U',
    # 'V',
    # 'W',
]

with open(save_file_name, 'w') as fp:
    fp.write('\\begin{tabular}{l|' + len(col_names)*'r|' + '}\n')
    fp.write('\\hline\n')
    # fp.write('& \\multicolumn{{{}}}{{c|}}{{{}}}\\\\ \n'.format(
    #     (ncomps+use_bg)*2,
    #     'Uniform Background',
    # ))
    # for i in range(ncomps):
    #     fp.write('& \\multicolumn{{2}}{{c |}}{{Component {}}}'.format(chr(65+i)))
    for col_name in col_names:
        fp.write('{} & '.format(col_name.replace('_',' ')))
    fp.write('Membership prob \\\\\n')
    fp.write('\\hline\n')

    for ix, row in enumerate(star_pars['table'][bpmg_table_mask]):
        if np.isnan(row['radial_velocity_best']):
            row['radial_velocity_best'] = row['radial_velocity']
            row['radial_velocity_error_best'] = row['radial_velocity_error']
            row['radial_velocity_best_flag'] = 'Gaia'
        fp.write('{} '.format(row[col_names[0]]))
        for col_name in col_names[1:]:
            if type(row[col_name]) is np.float64:
                fp.write('& {:5.3f} '.format(row[col_name]))
            else:
                fp.write('& {:10} '.format(row[col_name]))
        fp.write('& {:.2f} '.format(bpmg_memb_probs[np.where(bpmg_memb_probs > threshold)][ix]))
        fp.write('\\\\\n')

    fp.write('\\hline\n')
    fp.write('\\end{tabular}\n')

