from __future__ import print_function, division

"""
Generates a basic latex table from origin Synthetic.Group object(s)
and final median and errs file
"""

import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.datatool as dt

save_file_name = '../results/tables/synth_bpmg_table.tex'

rdir = '../results/em_fit/synth_bpmg/'
origin_file = rdir + 'synth_data/origins.npy'
fitted_z_file = rdir + 'final_membership.npy'
med_errs_file = rdir + 'final_med_errs.npy'

star_pars_file = '../data/synth_bpmg_xyzuvw.fits'

origins = dt.loadGroups(origin_file)
true_z = dt.getZfromOrigins(origins, star_pars_file)
true_nstars = np.sum(true_z, axis=0)
fitted_z = np.load(fitted_z_file)
fitted_nstars = np.sum(fitted_z, axis=0)
med_errs = np.load(med_errs_file)
med_errs[:,6:8] = np.exp(med_errs[:,6:8])

# manually work out ordering of groups
origins = origins[::-1]
true_nstars = true_nstars[([1,0,2],)]
# manually inform on presence of background or not
use_bg = True

row_names = ['X [pc]', 'Y [pc]', 'Z [pc]',
             'U [$\kms$]', 'V [$\kms$]', 'W [$\kms$]',
             '$r_0$', '$\sigma_0$', 'age [Myr]', 'nstars']

ncomps = len(origins)# + use_bg

with open(save_file_name, 'w') as fp:
    fp.write('\\begin{tabular}{l|' + 2*(ncomps+use_bg)*'r|' + '}\n')
    fp.write('\\hline\n')
    fp.write('& \\multicolumn{{{}}}{{c|}}{{{}}}\\\\ \n'.format(
        (ncomps+use_bg)*2,
        'Uniform Background',
    ))
    for i in range(ncomps):
        fp.write('& \\multicolumn{{2}}{{c |}}{{Component {}}}'.format(chr(65+i)))
    if use_bg:
        fp.write('& \\multicolumn{2}{c |}{Background }')
    fp.write('\n\\\\\n')
    fp.write('& True & Fit' * (ncomps + use_bg) + '\\\\\n')
    fp.write('\\hline\n')

    for i, row_name in enumerate(row_names[:-1]):
        line = '{:11}'.format(row_name)
        for j in range(ncomps):
            line += ' & ${:5.1f}$'.format(origins[j].pars[i])
            line += ' & ${:5.1f}^{{+{:4.1f}}}_{{-{:4.1f}}}$ '.format(
                med_errs[j][i][0],
                med_errs[j][i][1] - med_errs[j][i][0],
                med_errs[j][i][0] - med_errs[j][i][2],
            )
        if use_bg:
            line += ' & - & -'
        line += '\\\\\n'
        fp.write(line)
    line = '{:11}'.format(row_names[-1])
    for j in range(ncomps):
        line += '& {:3} '.format(int(true_nstars[j]))
        line += '& {:4.2f} '.format(fitted_nstars[j])
    if use_bg:
        line += ' & {} & {:4.2f}'.format(true_nstars[-1], fitted_nstars[-1])
    line += ' \\\\\n'
    fp.write(line)
    fp.write('\\hline\n')
    fp.write('\\end{tabular}\n')


    # fp.write('& \\multicolumn\{{}\}\{c |\}\{{}\}\n'.format(ncomps*3, 'synth bpmg'))
    # for comp_ix in range(ncomps):
    #     &  \multicolumn
    #     {2}
    #     {c |}
    #     {Component
    #     A}
    #     &  \multicolumn
    #     {2}
    #     {c |}
    #     {Component
    #     B}
    #     &  \multicolumn
    #     {2}
    #     {c |}
    #     {Component
    #     C}
