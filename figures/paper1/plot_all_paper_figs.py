from __future__ import print_function, division

"""
A script which gathers all plotting of all relevant figures into
one spot to facilitate quick and simple replotting as needed.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../..')

import chronostar.datatool as dt
import chronostar.synthesiser as syn
import chronostar.fitplotter as fp
import chronostar.traceorbit as torb

debugging_circles=True

# PLOTTING FLAGS
PLOT_FED_STARS = False
# PLOT_FED_STARS = True
PLOT_MUTLI_SYNTH = False
# PLOT_MUTLI_SYNTH = True
# PLOT_BPMG_REAL = False
PLOT_BPMG_REAL = True
# PLOT_BANYAN_BPMG = False  # I don't think we need this one anymore
# PLOT_BANYAN_BPMG = True

LABELS = 'xyzuvw'

# if PLOT_BANYAN_BPMG:
#     fit_name = 'banyan_bpmg'
#     rdir = '../../results/em_fit/beta_Pictoris/'
#
#     memb_file = rdir + 'final_membership.npy'
#     groups_file = rdir + 'final_best_groups.npy'
#     star_pars_file = '../../data/beta_Pictoris_with_gaia_small_xyzuvw.fits'
#
#     z = np.load(memb_file)
#     groups = dt.loadGroups(groups_file)
#     star_pars = dt.loadDictFromTable(star_pars_file, 'beta Pictoris')
#
#     # First do all, then just do possible membs of BPMG
#     for dim1, dim2 in [(0,1)]: #, (0, 3), (1, 4), (2, 5)]:  # , 'yv', 'zw']:
#         plt.clf()
#         range_1 = [
#             np.min(star_pars['xyzuvw'][:, dim1]),
#             np.max(star_pars['xyzuvw'][:, dim1]),
#         ]
#         buffer = 0.1 * (range_1[1] - range_1[0])
#         range_1[0] -= buffer
#         range_1[1] += buffer
#         range_2 = [
#             np.min(star_pars['xyzuvw'][:, dim2]),
#             np.max(star_pars['xyzuvw'][:, dim2]),
#         ]
#         buffer = 0.1 * (range_2[1] - range_2[0])
#         range_2[0] -= buffer
#         range_2[1] += buffer
#         fp.plotPaneWithHists(
#             dim1,
#             dim2,
#             groups=groups,
#             star_pars=star_pars,
#             group_now=True,
#             membership=z,
#             # true_memb=true_memb,
#             savefile='{}_{}{}.pdf'.format(fit_name, LABELS[dim1],
#                                           LABELS[dim2]),
#             with_bg=True,
#             range_1=range_1,
#             range_2=range_2,
#         )

if PLOT_BPMG_REAL:
    fit_name = 'bpmg_and_nearby'
    rdir = '../../results/em_fit/beta_Pictoris_wgs_inv2_6C_res/'

    memb_file = rdir + 'final_membership.npy'
    groups_file = rdir + 'final_groups.npy'
    star_pars_file = '../../data/beta_Pictoris_with_gaia_small_xyzuvw.fits'

    z = np.load(memb_file)
    star_pars = dt.loadXYZUVW(star_pars_file)
    groups = dt.loadGroups(groups_file)

    # Assign markers based on BANYAN membership
    gt_sp = dt.loadDictFromTable('../../data/banyan_with_gaia_near_bpmg_xyzuvw.fits')
    banyan_membership = len(star_pars['xyzuvw']) * ['N/A']
    for i in range(len(star_pars['xyzuvw'])):
        master_table_ix = np.where(gt_sp['table']['source_id']==star_pars['gaia_ids'][i])
        banyan_membership[i] = gt_sp['table']['Moving group'][master_table_ix[0][0]]


    # assign markers based on present moving groups, keep track of
    # assoc -> marker relationship incase a legend is called for
    banyan_membership=np.array(banyan_membership)
    banyan_markers = np.array(len(banyan_membership) * ['.'])

    banyan_memb_set = set(banyan_membership)
    banyan_markers[np.where(banyan_membership=='beta Pictoris')] = 'v'
    marker_label = []
    banyan_memb_set.remove('beta Pictoris')
    marker_label.append('beta Pictoris')
    marker_style = []
    marker_style.append('v')
    banyan_markers[np.where(banyan_membership=='Tucana-Horologium')] = '*'
    banyan_memb_set.remove('Tucana-Horologium')
    marker_label.append('Tucana-Horologium')
    marker_style.append('*')

    banyan_memb_set.remove('N/A')
    for banyan_assoc, marker in zip(banyan_memb_set, ('s', 'p', 'D', 'X', 'H', 'D')): #''''''^', '<', '>', '8', 's', 'p', 'h', 'H', 'D', 'd', 'P', 'X')):
        banyan_markers[np.where(banyan_membership==banyan_assoc)] = marker
        marker_label.append(banyan_assoc)
        marker_style.append(marker)

    # First do all, then just do possible membs of BPMG
    if True:
        nearby_range = {}
        for dim in range(6):
            nearby_range[dim] = [
                np.min(star_pars['xyzuvw'][:, dim]),
                np.max(star_pars['xyzuvw'][:, dim]),
            ]
            buffer = 0.1 * (nearby_range[dim][1] - nearby_range[dim][0])
            nearby_range[dim][0] -= buffer
            nearby_range[dim][1] += buffer
            # Temporarily fix y to be that of x so cirlces are clearly circles
        if debugging_circles:
            nearby_range[1] = [-120, 80]

        for dim1, dim2 in [(0,1), (0,3), (1,4), (2,5)]: #, 'yv', 'zw']:
            plt.clf()
            range_1 = [
                np.min(star_pars['xyzuvw'][:,dim1]),
                np.max(star_pars['xyzuvw'][:,dim1]),
            ]
            buffer = 0.1 * (range_1[1] - range_1[0])
            range_1[0] -= buffer
            range_1[1] += buffer
            range_2 = [
                np.min(star_pars['xyzuvw'][:,dim2]),
                np.max(star_pars['xyzuvw'][:,dim2]),
            ]
            buffer = 0.1 * (range_2[1] - range_2[0])
            range_2[0] -= buffer
            range_2[1] += buffer
            fp.plotPaneWithHists(
                dim1,
                dim2,
                groups=groups,
                star_pars=star_pars,
                group_now=True,
                membership=z,
                # true_memb=true_memb,
                savefile='{}_{}{}.pdf'.format(fit_name, LABELS[dim1], LABELS[dim2]),
                with_bg=True,
                range_1=nearby_range[dim1], #range_1,
                range_2=nearby_range[dim2], #range_2,
                residual=True,
                markers=banyan_markers,
            )

    # Only include stars that, if they weren't bg, they'd most likely be BPMG
    if True:
        fit_name = 'bpmg_candidates'
        extract_group_ix = [0,2]
        # bpmg_mask = np.where(z[:,extract_group_ix]>0.1)
        bpmg_mask = np.where(np.isin(np.argmax(z[:,:-1], axis=1), extract_group_ix))# == extract_group_ix)
        star_pars['xyzuvw'] = star_pars['xyzuvw'][bpmg_mask]
        star_pars['xyzuvw_cov'] = star_pars['xyzuvw_cov'][bpmg_mask]
        star_pars['indices'] = np.array(star_pars['indices'])[bpmg_mask]
        z = z[bpmg_mask]#, (0,-1),]
        z = z[:,(extract_group_ix+[-1]),]

        bpmg_range = {}
        for dim in range(6):
            bpmg_range[dim] = [
                np.min(star_pars['xyzuvw'][:, dim]),
                np.max(star_pars['xyzuvw'][:, dim]),
            ]
            buffer = 0.1 * (bpmg_range[dim][1] - bpmg_range[dim][0])
            bpmg_range[dim][0] -= buffer
            bpmg_range[dim][1] += buffer
        # Temporarily fix y to be that of x so cirlces are clearly circles
        if debugging_circles:
            bpmg_range[1] = [-120,80] # DELETE THIS!!!

        for dim1, dim2 in [(0,1), (0,3), (1,4)]: #, (2,5)]: #, 'yv', 'zw']:
            fp.plotPaneWithHists(
                dim1,
                dim2,
                groups=groups[extract_group_ix],
                star_pars=star_pars,
                group_now=True,
                membership=z,
                savefile='{}_{}{}.pdf'.format(fit_name,
                                                      LABELS[dim1],
                                                      LABELS[dim2]),
                with_bg=True,
                range_1=bpmg_range[dim1],
                range_2=bpmg_range[dim2],
                residual=True,
                markers=banyan_markers,
            )

        # To ensure consistency, we now plot the BANYAN bpmg stars only,
        # and use the ragnes from previous plot
        fit_name = 'banyan_bpmg'
        rdir = '../../results/em_fit/beta_Pictoris/'

        memb_file = rdir + 'final_membership.npy'
        groups_file = rdir + 'final_best_groups.npy'
        star_pars_file = '../../data/beta_Pictoris_with_gaia_small_xyzuvw.fits'

        z = np.load(memb_file)
        groups = dt.loadGroups(groups_file)
        star_pars = dt.loadDictFromTable(star_pars_file, 'beta Pictoris')

        # First do all, then just do possible membs of BPMG
        for dim1, dim2 in [(0,1), (0, 3), (1, 4)]: #, (2, 5)]:  # , 'yv', 'zw']:
            fp.plotPaneWithHists(
                dim1,
                dim2,
                groups=groups,
                star_pars=star_pars,
                group_now=True,
                membership=z,
                # true_memb=true_memb,
                savefile='{}_{}{}.pdf'.format(fit_name, LABELS[dim1],
                                              LABELS[dim2]),
                with_bg=True,
                range_1=bpmg_range[dim1],
                range_2=bpmg_range[dim2],
                markers=banyan_markers,
            )


# plotting federrath stars
if PLOT_FED_STARS:
    synth_fit = 'fed_stars'
    rdir = '../../results/fed_fits/30/gaia/'
    # rdir = '../../results/new_fed_stars_20/gaia/'
    origins_file = rdir + 'origins.npy'
    chain_file = rdir + 'final_chain.npy'
    lnprob_file = rdir + 'final_lnprob.npy'
    star_pars_file = rdir + 'xyzuvw_now.fits'
    # init_xyzuvw_file = '../../data/sink_init_xyzuvw.npy'
    init_xyzuvw_file = rdir + '../xyzuvw_init_offset.npy'
    # perf_xyzuvw_file = rdir + '../perf_xyzuvw.npy'
    # star_pars_file = '../../data/fed_stars_20_xyzuvw.fits'

    chain = np.load(chain_file).reshape(-1,9)
    lnprobs = np.load(lnprob_file)
    # best_fit_pars = np.load(chain_file)[np.unravel_index(np.argmax(lnprobs), lnprobs.shape)]
    best_fit_pars = chain[np.argmax(lnprobs)]
    groups = [syn.Group(best_fit_pars, internal=True, starcount=False)]
    origins = dt.loadGroups(origins_file)
    raw_init_xyzuvw = np.load(init_xyzuvw_file)
    # perf_xyzuvw = np.load(perf_xyzuvw_file)
    # init_xyzuvw = torb.traceManyOrbitXYZUVW(perf_xyzuvw, -origins[0].age,
    #                                         single_age=True)
    init_xyzuvw = np.load(init_xyzuvw_file)

    for dim1, dim2 in ['xy', 'xu', 'yv']:
        plt.clf()
        fp.plotPane(
            dim1,
            dim2,
            groups=groups,
            star_pars={'xyzuvw':init_xyzuvw},
            group_then=True,
            savefile='{}_then_{}{}.pdf'.format(synth_fit, dim1, dim2)
        )
        plt.tight_layout(pad=0.7)


    for dim1, dim2 in ['xy', 'xu', 'yv']:
        fp.plotPaneWithHists(
            dim1,
            dim2,
            groups=groups,
            star_pars=star_pars_file,
            group_now=True,
            group_then=True,
            star_orbits=True,
            group_orbit=True,
            membership=None,
            true_memb=None,
            savefile='{}_now_{}{}.pdf'.format(synth_fit, dim1, dim2)
        )

# plotting Multi-component synth fits
if PLOT_MUTLI_SYNTH:
    synth_fits = [
        'synth_bpmg',
        'four_assocs',
        'assoc_in_field',
        'same_centroid',
    ]

    rdir_suffix = {
        'synth_bpmg':'',
        'four_assocs':'_res',
        'assoc_in_field':'_res',
        'same_centroid':'_res',
    }

    planes = {
        'synth_bpmg':['xu', 'zw', 'xy', 'yz'],
        'four_assocs':['xy', 'yv'],
        'assoc_in_field':['uv', 'xu'],
        'same_centroid':['xu', 'yv'],
    }

    with_bg = {
        'synth_bpmg':True,
        'four_assocs':False,
        'assoc_in_field':True,
        'same_centroid':False,
    }

    for synth_fit in synth_fits:
        rdir = '../../results/em_fit/{}{}/'.format(synth_fit,
                                                   rdir_suffix[synth_fit])
        groups_file = rdir + 'final_best_groups.npy'
        # star_pars_file = rdir + '{}_xyzuvw.fits'.format(synth_fit)
        star_pars_file = '../../data/{}_xyzuvw.fits'.format(synth_fit)
        memb_file = rdir + 'final_membership.npy'
        origins_file = rdir + 'synth_data/origins.npy'
        true_memb = dt.getZfromOrigins(origins_file, star_pars_file)
        for dim1, dim2 in planes[synth_fit]:
            fp.plotPaneWithHists(
                dim1,
                dim2,
                groups=groups_file,
                star_pars=star_pars_file,
                group_now=True,
                membership=memb_file,
                true_memb=true_memb,
                savefile='{}_{}{}.pdf'.format(synth_fit, dim1, dim2),
                with_bg=with_bg[synth_fit],
                group_bg=(synth_fit == 'assoc_in_field')
            )

