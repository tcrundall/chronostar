from __future__ import print_function, division

"""
Takes the chains of fit to BPMG origin, and converts to chains of
parametrisation of corresponding current-day fits to BPMG
"""

import numpy as np
import sys
sys.path.insert(0, '..')
import chronostar.synthesiser as syn
import chronostar.transform as tf
import chronostar.traceorbit as torb
import chronostar.datatool as dt

DIM = 6

chaindirs = [
    # '../results/em_fit/beta_Pic_solo_results/',
    # '../results/em_fit/beta_Pic_results/group0/',
    # '../results/em_fit/beta_Pictoris_wgs_inv2_5B_res/',
    '../results/em_fit/beta_Pictoris_wgs_inv2_5B_tuc-hor_res/',
]

labels = [
    # 'solo',
    # 'near_gaia',
    '5comp',
]

cd_chains = {}
med_and_spans = {}

for chaindir, label in zip(chaindirs, labels):
    try:
        cd_chains[label] = np.load(chaindir + 'current_day_chain.npy')
    except IOError:
        origin_chain = np.load(chaindir + 'final_chain.npy').reshape(-1,9)

        npars = 6 + 6 + 15 # + 1 (don't need the age in current day fit)
        current_day_chain = np.zeros((origin_chain.shape[0],npars))

        for sample_ix, sample in enumerate(origin_chain):
            if sample_ix % 100 == 0:
                print("Done {:6} of {}".format(sample_ix, current_day_chain.shape[0]))
            mean = sample[:6]
            group = syn.Group(sample, internal=True, starcount=False)

            cd_mean = torb.traceOrbitXYZUVW(group.mean, group.age)
            cd_cov = tf.transform_cov(group.generateSphericalCovMatrix(), torb.traceOrbitXYZUVW,
                                      group.mean, args=(group.age, True))

            # fill in cartesian mean
            current_day_chain[sample_ix,0:6] = cd_mean

            # fill in standard deviations
            cd_stds = np.sqrt(cd_cov[np.diag_indices(DIM)])
            current_day_chain[sample_ix,6:12] = cd_stds

            correl_matrix = cd_cov / cd_stds / cd_stds.reshape(DIM, 1)
            # fill in correlations
            for col_ix in range(15):
                current_day_chain[sample_ix,12+col_ix] = correl_matrix[
                    np.triu_indices(DIM, k=1)[0][col_ix],
                    np.triu_indices(DIM, k=1)[1][col_ix]
                ]

        np.save(chaindir + 'current_day_chain.npy', current_day_chain)
            # I think I can write above line as:
            # gt_row[col_name] = correl_matrix[np.triu_indices(dim,k=1)][col_ix]
        cd_chains[label] = current_day_chain
        med_and_spans[label] = dt.calcMedAndSpan(cd_chains[label])
        np.save(chaindir + 'cd_med_and_span.npy', med_and_spans[label])

