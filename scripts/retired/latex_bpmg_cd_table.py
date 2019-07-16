from __future__ import print_function, division

"""
Generate latex table of BPMG origin and current day fits,
for both solely BANYAN members fit and nearby Gaia stars fit
"""

import numpy as np

import sys
sys.path.insert(0, '../..')
from chronostar.component import SphereComponent, FreeComponent
from chronostar import compfitter


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
        string_template = ' & ${:6.' + prec + 'f}^{{+{:4.' + prec + 'f}}}_{{' \
                                                                    '-{:4.' +\
                          prec + 'f}}}$ '
        latex_form = string_template.format(
                entry[0], entry[1] - entry[0], entry[0] - entry[2],
        )
    else:
        string_template = ' & ${:6.' + prec + 'f}$ '
        latex_form = string_template.format(entry)
    return latex_form


chaindirs = [
    # '../results/em_fit/beta_Pic_solo_results/',
    # '../results/em_fit/beta_Pic_results/group0/',
    # '../results/em_fit/beta_Pictoris_wgs_inv2_5B_res/',
    # '../results/em_fit/beta_Pictoris_wgs_inv2_5B_tuc-hor_res/',
    '../../results/beta_Pictoris_with_gaia_small_inv2/6/E/final/group0/',
    '../../results/beta_Pictoris_with_gaia_small_inv2/6/E/final/group3/',
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



# zs = {}
origin_med_and_spans = {}
cd_med_and_spans = {}

cd_file_stem = 'cd_final_med_errs.npy'

rdir = '../../results/beta_Pictoris_with_gaia_small_inv2/6/E/final/'

for label in labels:
    chaindir = rdir + 'group{}/'.format(comp_ix[label])
    try:
        cd_med_and_spans[label] = np.load(chaindir + cd_file_stem)
        print('loaded for {}'.format(label))
    except IOError:
        print('calculating for {}'.format(label))
        # Convert chain of sampling origin to corresponding chain of current day
        flat_final_chain = np.load(chaindir + 'final_chain.npy').reshape(-1, 9)
        nsamples = len(flat_final_chain)

        # initialise empty array
        current_day_chain = np.zeros((nsamples, len(FreeComponent.PARAMETER_FORMAT)))

        # One by one, get equivalent pars of current day
        for ix, sample in enumerate(flat_final_chain):
            if ix % 100 == 0:
                print('{} of {} done'.format(ix, len(flat_final_chain)))
            comp = SphereComponent(emcee_pars=sample)
            cd_mean, cd_cov = comp.get_currentday_projection()
            cd_comp = FreeComponent(attributes={'mean':cd_mean,
                                                'covmatrix':cd_cov,
                                                'age':0.})
            current_day_chain[ix] = cd_comp.get_pars()

        cd_med_and_spans[label] = compfitter.calc_med_and_span(current_day_chain)
        np.save(chaindir + cd_file_stem, cd_med_and_spans[label])
    # cd_med_and_spans[label] = np.load(chaindir + 'cd_med_and_span.npy')
    # cd_med_and_spans[label][6:8] = np.exp(cd_med_and_spans[label][6:8])
    origin_med_and_spans[label] = np.load(rdir + 'final_med_errs.npy')[comp_ix[label]]
    origin_med_and_spans[label][6:8] = np.exp(origin_med_and_spans[label][6:8])

zs = np.load(rdir + 'final_membership.npy')

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

save_file_names = {
    'nocorrs':'../../results/tables/table5.tex',
    'withcorrs':'../../results/tables/table5_full.tex',
}

corr_flags = {
    'nocorrs':False,
    'withcorrs':True,
}

for corr_variant, save_file_name in save_file_names.items():
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



        # MEANS (X,Y,Z,U,V,W)
        for i, row_name in enumerate(row_names[:6]):
            line = '{:20}'.format(row_name)
            for label in labels:
                line += generateStringFromEntry(
                    origin_med_and_spans[label][i]
                )
                line += generateStringFromEntry(cd_med_and_spans[label][i])
            line += '\\\\\n'
            fp.write(line)

        # STANDARD DEVS (dX, dY, dZ, dU, dV, dW)
        for i, row_name in enumerate(row_names[6:12]):
            line = '{:20}'.format(row_name)
            for label in labels:
                line += generateStringFromEntry(
                    origin_med_and_spans[label][6+int(i/3)], prec=1
                )
                line += generateStringFromEntry(cd_med_and_spans[label][6+i], prec=1)
            line += '\\\\\n'
            fp.write(line)

        # CORRESLATIONS
        if corr_flags[corr_variant]:
            for i, row_name in enumerate(row_names[12:27]):
                line = '{:20}'.format(row_name)
                for label in labels:
                    line += '& 0 '
                    line += generateStringFromEntry(cd_med_and_spans[label][12+i], prec=3)
                line += '\\\\\n'
                fp.write(line)

        # AGES
        for i, row_name in enumerate(row_names[27:28]):
            line = '{:20}'.format(row_name)
            for label in labels:
                line += '& - '
                line += generateStringFromEntry(
                    origin_med_and_spans[label][-1], prec=1)
            line += '\\\\\n'
            fp.write(line)

        # STAR COUNTS
        for i, row_name in enumerate(row_names[28:29]):
            line = '{:20}'.format(row_name)
            for label in labels:
                line += '& - '
                nstars = zs[:,comp_ix[label]].sum()
                line += generateStringFromEntry(nstars, prec=1, no_uncert=True)
            line += '\\\\\n'
            fp.write(line)


        fp.write('\\hline\n')
        fp.write('\\end{tabular}\n')

