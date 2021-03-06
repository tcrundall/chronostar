from __future__ import print_function, division

"""
Generates a basic latex table from origin Synthetic.Group object(s)
and final median and errs file
"""

import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.retired2.datatool as dt

fits = [
    'synth_bpmg2',
    'same_centroid',
    # 'test_on_motley3', # this is right.... but slightly different for some reason
    # 'field_blind',
    'assoc_in_field',
    'four_assocs',
    # results/em_builder12 or maybe 13
]
#origin_stems = [
#    'synth_data/origins.npy',
#    'synth_data/origins.npy',
#    'synth_data/origins.npy',
#    'synth_data/origins.npy',
#]
#fitted_z_stems = [
#    'final_membership.npy',
#    'final_membership.npy',
#    'final_membership.npy',
#    'final_memberships.npy',
#]
#med_errs_stems = [
#    'final_med_errs.npy',
#    'final_med_errs.npy',
#    'final_med_errs.npy',
#    'final_med_errs.npy',
#]
# star_pars_files = [
#     '../data/synth_bpmg_xyzuvw.fits',
#     '../data/'
#     '../results/em_fit/test_on_motley3/xyzuvw_now.fits',
#     '../results/em_fit/field_blind/xyzuvw_now.fits',
#     None,
# ]

suffixs = ['_res', '_res', '_res', '_res']

orders = [
    [1,0],
    [1,0],
    [1,0],
    [3,2,0,1],
]

origin_stem = 'synth_data/origins.npy'
fitted_z_stem = 'final_membership.npy'
med_errs_stem = 'final_med_errs.npy'
use_bg = False

for fit, order, suffix in zip(fits, orders, suffixs):
    print(fit)
    rdir = '../results/em_fit/{}'.format(fit) + suffix + '/'

    star_pars_file = '../data/{}_xyzuvw.fits'.format(fit)
    save_file_name = '../results/tables/{}_table.tex'.format(fit)

    origin_file = rdir + origin_stem
    fitted_z_file = rdir + fitted_z_stem
    med_errs_file = rdir + med_errs_stem

    origins = dt.loadGroups(origin_file)
    true_z = dt.getZfromOrigins(origins, star_pars_file)
    true_nstars = np.sum(true_z, axis=0)
    fitted_z = np.load(fitted_z_file)
    fitted_nstars = np.sum(fitted_z, axis=0)
    med_errs = np.load(med_errs_file)
    med_errs[:,6:8] = np.exp(med_errs[:,6:8])

    # manually work out ordering of groups
    origins = origins[(order,)]
    if use_bg:
        true_nstars = true_nstars[(order + [-1],)]
    else:
        true_nstars = true_nstars[(order,)]
    # manually inform on presence of background or not

    row_names = ['$x_0$ [pc]', '$y_0$ [pc]', '$z_0$ [pc]',
                 '$u_0$ [$\kms$]', '$v_0$ [$\kms$]', '$w_0$ [$\kms$]',
                 '$\sigma_{xyz}$ [pc]', '$\sigma_{uvw} [\kms]$', 'age [Myr]', 'nstars']

    ncomps = len(origins)# + use_bg

    with open(save_file_name, 'w') as fp:
        fp.write('\\begin{tabular}{l|' + 2*(ncomps+use_bg)*'r|' + '}\n')
        fp.write('\\hline\n')
        # fp.write('& \\multicolumn{{{}}}{{c|}}{{{}}}\\\\ \n'.format(
        #     (ncomps+use_bg)*2,
        #     'Uniform Background',
        # ))
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
                line += ' & ${:5.1f}$'.format(origins[j].getSphericalPars()[i])
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

