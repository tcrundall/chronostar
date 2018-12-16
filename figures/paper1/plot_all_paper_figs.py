from __future__ import print_function, division

"""
A script which gathers all plotting of all relevant figures into
one spot to facilitate quick and simple replotting as needed.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../..')

import chronostar.fitplotter as fp
import chronostar.datatool as dt
import chronostar.synthesiser as syn
import chronostar.traceorbit as torb

# PLOTTING FLAGS
# PLOT_FED_STARS = False
PLOT_FED_STARS = True
PLOT_MUTLI_SYNTH = False
# PLOT_MUTLI_SYNTH = True
# PLOT_BPMG_REAL = False
PLOT_BPMG_REAL = True

LABELS = 'xyzuvw'


if PLOT_BPMG_REAL:
    fit_name = 'bpmg_and_nearby'
    rdir = '../../results/em_fit/beta_Pictoris_wgs_inv2_6C_res/'

    memb_file = rdir + 'final_membership.npy'
    groups_file = rdir + 'final_groups.npy'
    star_pars_file = '../../data/beta_Pictoris_with_gaia_small_xyzuvw.fits'

    z = np.load(memb_file)
    # bpmg_mask = np.where(z[:,4]>0.01)

    star_pars = dt.loadXYZUVW(star_pars_file)
    # star_pars['xyzuvw'] = star_pars['xyzuvw'][bpmg_mask]
    # star_pars['xyzuvw_cov'] = star_pars['xyzuvw_cov'][bpmg_mask]
    # star_pars['indices'] = np.array(star_pars['indices'])[bpmg_mask]

    groups = dt.loadGroups(groups_file)
    # groups = groups[4:5]

    # getting BANYAN intersections:
    if False:
        comp_ids = {}
        for i in range(7):
            comp_ids[i] = star_pars['gaia_ids'][np.where(z[:,i] > 0.5)]

        gt_sp = dt.loadDictFromTable('../../data/banyan_with_gaia_near_bpmg_xyzuvw.fits')
        for el in gt_sp['table']['Moving group'][np.where(np.isin(gt_sp['table']['source_id'], comp_ids[4]))]:
            print(el)

    # import pdb; pdb.set_trace()
    # First do all, then just do possible membs of BPMG
    if False:
        for dim1, dim2 in [(0,3), (1,4), (2,5)]: #, 'yv', 'zw']:
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
                # group_then=True,
                # star_orbits=True,
                # group_orbit=True,
                membership=z,#[:,(4,0,1,2,3,5,-1)],
                # true_memb=true_memb,
                savefile='{}_{}{}.pdf'.format(fit_name, LABELS[dim1], LABELS[dim2]),
                with_bg=True,
                range_1=range_1,
                range_2=range_2,
                residual=True,
            )
            # plt.savefig('{}_{}{}.pdf'.format(synth_fit, dim1, dim2))
    if True:
        fit_name = 'bpmg_candidates'
        extract_group_ix = 0
        # bpmg_mask = np.where(z[:,extract_group_ix]>0.1)
        bpmg_mask = np.where(np.argmax(z[:,:-1], axis=1) == extract_group_ix)
        star_pars['xyzuvw'] = star_pars['xyzuvw'][bpmg_mask]
        star_pars['xyzuvw_cov'] = star_pars['xyzuvw_cov'][bpmg_mask]
        star_pars['indices'] = np.array(star_pars['indices'])[bpmg_mask]
        z = z[bpmg_mask]#, (0,-1),]
        z = z[:,(extract_group_ix,-1),]
        for dim1, dim2 in [(0,3), (1,4), (2,5)]: #, 'yv', 'zw']:
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
            # import pdb; pdb.set_trace()
            fp.plotPaneWithHists(
                dim1,
                dim2,
                groups=[groups[extract_group_ix]],
                star_pars=star_pars,
                group_now=True,
                # group_then=True,
                # star_orbits=True,
                # group_orbit=True,
                membership=z,#[:,(4,0,1,2,3,5,-1)],
                # true_memb=true_memb,
                # savefile='{}_{}{}_with_bg.pdf'.format(synth_fit, dim1, dim2),
                with_bg=True,
                range_1=range_1,
                range_2=range_2,
            )
            plt.savefig('{}_{}{}.pdf'.format(fit_name, LABELS[dim1], LABELS[dim2]))


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
        'four_assocs',
        'assoc_in_field',
        'same_centroid',
    ]

    planes = {
        'four_assocs':['xy', 'yv'],
        'assoc_in_field':['uv', 'xu'],
        'same_centroid':['xu', 'yv'],
    }

    for synth_fit in synth_fits:
        rdir = '../../results/em_fit/{}_res/'.format(synth_fit)
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
                # group_then=True,
                # star_orbits=True,
                # group_orbit=True,
                membership=memb_file,
                true_memb=true_memb,
                savefile='{}_{}{}.pdf'.format(synth_fit, dim1, dim2)
            )

