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
    # '../results/em_fit/beta_Pic_solo_results/',
    # '../results/em_fit/beta_Pic_results/group0/',
    '../results/em_fit/beta_Pictoris_wgs_inv2_5B_res/',
    '../results/em_fit/beta_Pictoris_wgs_inv2_5B_tuc-hor_res/',
]
labels = [
    # 'solo',
    # 'near_gaia',
    'bpmg',
    'tuc-hor',
]

comp_ix = {
    'bpmg':0,
    'tuc-hor':3,
}

zs = {}
origin_med_and_spans = {}
cd_med_and_spans = {}

for chaindir, label in zip(chaindirs, labels):
    cd_med_and_spans[label] = np.load(chaindir + 'cd_med_and_span.npy')
    # cd_med_and_spans[label][6:8] = np.exp(cd_med_and_spans[label][6:8])
    origin_med_and_spans[label] = np.load(chaindir + 'final_med_errs.npy')
    origin_med_and_spans[label][:,6:8] = np.exp(origin_med_and_spans[label][:,6:8])
    zs[label] = np.load(chaindir + '/final_membership.npy')



# import pdb; pdb.set_trace()

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
    'nstars',
]

lb_len = max([len(label) for label in row_names])

nruns = len(chaindirs)

with open(save_file_name, 'w') as fp:
    fp.write('\\begin{tabular}{l' + 2*(nruns)*'|r' + '}\n')
    fp.write('\\hline\n')
    fp.write('& \\multicolumn{{{}}}{{c|}}{{{}}}'.format(
        2,
        r'$\beta$PMG',
    ))
    if nruns == 2:
        fp.write('& \\multicolumn{{{}}}{{c|}}{{{}}}'.format(
            2,
            'Partial Tuc-Hor',
        ))
    fp.write('\\\\\n')

    fp.write('& Origin & Current' * (nruns) + '\\\\\n')
    fp.write('\\hline\n')

    def generateStringFromEntry(entry, prec=1, no_uncert=False):
        """
        Entry is a list(like) of three values. The 50th, 84th and 16th percentile
        of a parameter.
        """
        prec = str(prec)
        if not no_uncert:
            # increase precision until uncertainties have at least 1 sig fig
            while float(('{:.' + prec + 'f}').format(entry[1] - entry[0])) == 0.0:
                prec = str(int(prec) + 1)
            string_template = ' & ${:6.' + prec + 'f}^{{+{:4.'+prec+'f}}}_{{-{:4.'+prec+'f}}}$ '
            latex_form = string_template.format(
                    entry[0], entry[1] - entry[0], entry[0] - entry[2],
                )
        else:
            string_template = ' & ${:6.' + prec + 'f}$ '
            latex_form = string_template.format(entry)
        return latex_form

    for i, row_name in enumerate(row_names[:6]):
        line = '{:20}'.format(row_name)
        for label in labels:
            line += generateStringFromEntry(
                origin_med_and_spans[label][comp_ix[label]][i]
            )
            line += generateStringFromEntry(cd_med_and_spans[label][i])
        line += '\\\\\n'
        fp.write(line)

    for i, row_name in enumerate(row_names[6:12]):
        line = '{:20}'.format(row_name)
        for label in labels:
            line += generateStringFromEntry(
                origin_med_and_spans[label][comp_ix[label]][6+int(i/3)], prec=1
            )
            line += generateStringFromEntry(cd_med_and_spans[label][6+i], prec=1)
        line += '\\\\\n'
        fp.write(line)

    corrs = False
    if corrs:
        for i, row_name in enumerate(row_names[12:27]):
            line = '{:20}'.format(row_name)
            for label in labels:
                line += '& 0 '
                line += generateStringFromEntry(cd_med_and_spans[label][12+i], prec=3)
            line += '\\\\\n'
            fp.write(line)

    for i, row_name in enumerate(row_names[27:28]):
        line = '{:20}'.format(row_name)
        for label in labels:
            line += '& 0 '
            line += generateStringFromEntry(
                origin_med_and_spans[label][comp_ix[label],-1], prec=1)
        line += '\\\\\n'
        fp.write(line)

    for i, row_name in enumerate(row_names[28:29]):
        line = '{:20}'.format(row_name)
        for label in labels:
            line += '& 0 '
            nstars = zs[label][:,comp_ix[label]].sum()
            line += generateStringFromEntry(nstars, prec=1, no_uncert=True)
        line += '\\\\\n'
        fp.write(line)


    fp.write('\\hline\n')
    fp.write('\\end{tabular}\n')

