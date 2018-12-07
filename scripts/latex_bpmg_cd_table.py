from __future__ import print_function, division

"""
Generate latex table of BPMG origin and current day fits,
for both solely BANYAN members fit and nearby Gaia stars fit
"""

import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.datatool as dt

save_file_name = '../results/tables/real_bpmg_table.tex'

chaindirs = [
    '../results/em_fit/beta_Pic_solo_results/',
    '../results/em_fit/beta_Pic_results/group0/',
]
labels = [
    'solo',
    'near_gaia',
]

origin_med_and_spans = {}
cd_med_and_spans = {}

for chaindir, label in zip(chaindirs, labels):
    cd_med_and_spans[label] = np.load(chaindir + 'cd_med_and_span.npy')
    origin_med_and_spans[label] = np.load(chaindir + 'final_med_errs.npy')

row_names = [
    'x [pc]',
    'y [pc]',
    'z [pc]',
    'u [$\kms$]',
    'v [$\kms$]',
    'w [$\kms$]',
    '$\sigma_x$ [pc]',
    '$\sigma_y$ [pc]',
    '$\sigma_z$ [pc]',
    '$\sigma_u [\kms]$',
    '$\sigma_v [\kms]$',
    '$\sigma_w [\kms]$',
    'corr$_{xy}$',
    'corr$_{xz}$',
    'corr$_{xu}$',
    'corr$_{xv}$',
    'corr$_{xw}$',
    'corr$_{yz}$',
    'corr$_{yu}$',
    'corr$_{yv}$',
    'corr$_{yw}$',
    'corr$_{zu}$',
    'corr$_{zv}$',
    'corr$_{zw}$',
    'corr$_{uv}$',
    'corr$_{uw}$',
    'corr$_{vw}$',
    'age [Myr]',
    'nstars']

lb_len = max([len(label) for label in row_names])

nruns = len(chaindirs)

with open(save_file_name, 'w') as fp:
    fp.write('\\begin{tabular}{l' + 2*(nruns)*'|r' + '}\n')
    fp.write('\\hline\n')
    fp.write('& \\multicolumn{{{}}}{{c|}}{{{}}}'.format(
        2,
        'BANYAN $\\beta$PMG members',
    ))
    if nruns == 2:
        fp.write('& \\multicolumn{{{}}}{{c|}}{{{}}}'.format(
            2,
            'BANYAN $\\beta$PMG members plus nearby \\textit{Gaia}',
        ))
    fp.write('\\\\\n')

    fp.write('& Origin & Current Day' * (nruns) + '\\\\\n')
    fp.write('\\hline\n')

    def generateStringFromEntry(entry, prec=1):
        """
        Entry is a list(like) of three values. The 50th, 84th and 16th percentile
        of a parameter.
        """
        prec = str(prec)
        string_template = ' & ${:6.' + prec + 'f}^{{+{:4.'+prec+'f}}}_{{-{:4.'+prec+'f}}}$ '
        latex_form = string_template.format(
                entry[0], entry[1] - entry[0], entry[0] - entry[2],
            )
        return latex_form

    for i, row_name in enumerate(row_names[:6]):
        line = '{:20}'.format(row_name)
        for label in labels:
            line += generateStringFromEntry(origin_med_and_spans[label][0][i])
            line += generateStringFromEntry(cd_med_and_spans[label][i])
        line += '\\\\\n'
        fp.write(line)

    for i, row_name in enumerate(row_names[6:12]):
        line = '{:20}'.format(row_name)
        for label in labels:
            line += generateStringFromEntry(origin_med_and_spans[label][0][6+int(i/3)])
            line += generateStringFromEntry(cd_med_and_spans[label][6+i])
        line += '\\\\\n'
        fp.write(line)

    for i, row_name in enumerate(row_names[12:27]):
        line = '{:20}'.format(row_name)
        for label in labels:
            line += '& 0 '
            line += generateStringFromEntry(cd_med_and_spans[label][12+i], prec=3)
        line += '\\\\\n'
        fp.write(line)

    #
    # for i, row_name in enumerate(row_names[:-1]):
    #     line = '{:11}'.format(row_name)
    #     for j in range(ncomps):
    #         line += ' & ${:5.1f}$'.format(origins[j].pars[i])
    #         line += ' & ${:5.1f}^{{+{:4.1f}}}_{{-{:4.1f}}}$ '.format(
    #             med_errs[j][i][0],
    #             med_errs[j][i][1] - med_errs[j][i][0],
    #             med_errs[j][i][0] - med_errs[j][i][2],
    #         )
    #     if use_bg:
    #         line += ' & - & -'
    #     line += '\\\\\n'
    #     fp.write(line)
    # line = '{:11}'.format(row_names[-1])
    # for j in range(ncomps):
    #     line += '& {:3} '.format(int(true_nstars[j]))
    #     line += '& {:4.2f} '.format(fitted_nstars[j])
    # if use_bg:
    #     line += ' & {} & {:4.2f}'.format(true_nstars[-1], fitted_nstars[-1])
    # line += ' \\\\\n'
    # fp.write(line)
    fp.write('\\hline\n')
    fp.write('\\end{tabular}\n')
    #
    #
    # # fp.write('& \\multicolumn\{{}\}\{c |\}\{{}\}\n'.format(ncomps*3, 'synth bpmg'))
    # # for comp_ix in range(ncomps):
    # #     &  \multicolumn
    # #     {2}
    # #     {c |}
    # #     {Component
    # #     A}
    # #     &  \multicolumn
    # #     {2}
    # #     {c |}
    # #     {Component
    # #     B}
    # #     &  \multicolumn
    # #     {2}
    # #     {c |}
    # #     {Component
    # #     C}
