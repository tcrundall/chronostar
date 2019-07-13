from __future__ import print_function, division

import chronostar.component

"""
A script which gathers all plotting of all relevant figures into
one spot to facilitate quick and simple replotting as needed.

TODO: Maybe do again, but with the range buffer lowered to 0.1 (from 0.2)
"""

import corner
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../..')

import chronostar.retired2.datatool as dt
import chronostar.fitplotter as fp
from chronostar import tabletool
from chronostar.component import SphereComponent, FreeComponent, EllipComponent
from astropy.table import Table

debugging_circles=False

# PLOTTING FLAGS
PLOT_CORNER = False
# PLOT_CORNER = True
PLOT_FED_STARS = False
# PLOT_FED_STARS = True
PLOT_MUTLI_SYNTH = False
# PLOT_MUTLI_SYNTH = True
PLOT_SYNTH_BPMG2 = False
# PLOT_SYNTH_BPMG2 = True
# PLOT_BPMG_REAL = False
PLOT_BPMG_REAL = True
PLOT_FAILURE = False

DEFAULT_DIMS = ((0,1), (0,3), (1,4), (2,5))
COLOR_LABELS = ['Fitted {}'.format(ch) for ch in 'ABCDEF']
COLORS = [
    'xkcd:blue',
    'xkcd:red',
    'xkcd:tangerine',
    'xkcd:shit',
    'xkcd:neon purple',
    'xkcd:sun yellow',
    'xkcd:cyan',
    'xkcd:bright pink'
]

acronyms = {
    'beta Pictoris':r'$\beta$PMG',
    'Tucana-Horologium':'Tuc-Hor',
    # 'Columba':'Columba',
    # 'Carina':'CAR',
    # 'TW Hya':'TWA',
    'Upper Centaurus Lupus':'UCL',
}

labels = 'XYZUVW'
units = 3 * ['[pc]'] + 3 * ['[km/s]']


def displayRanges(ranges):
    print([ranges[dim][1] - ranges[dim][0] for dim in (0,1,2,3,4,5)])

def calcRanges(star_pars, sep_axes=False, scale=True):
    """Simple function to calculate span in each dimension of stars with
    10% buffer"""
    ranges = {}
    for dim in range(star_pars['xyzuvw'].shape[1]):
        ranges[dim] = [
            np.min(star_pars['xyzuvw'][:, dim]),
            np.max(star_pars['xyzuvw'][:, dim]),
        ]
        buffer = 0.05 * (ranges[dim][1] - ranges[dim][0])
        ranges[dim][0] -= buffer
        ranges[dim][1] += buffer

    # adjust ranges so the span is consistent across pos axes and vel axes

    for dim in (3,4,5):
        print(ranges[dim][1] - ranges[dim][0])

    if sep_axes:
        xranges = {}
        yranges = {}
        for key in ranges.keys():
            xranges[key] = ranges[key][:]
            yranges[key] = ranges[key][:]
        if scale:
            scaleRanges(xranges, dims=(0,1,2))
            scaleRanges(xranges, dims=(3,4,5))
            scaleRanges(yranges, dims=(0,1,2))
            scaleRanges(yranges, dims=(3,4,5))
        return xranges, yranges
    else:
        if scale:
            scaleRanges(ranges, dims=(0,1,2))
            scaleRanges(ranges, dims=(3,4,5))

        return ranges


def scaleRanges(ranges, dims=(0,1,2)):
    """
    Rescale elements (inplace) in range such that span is equivalent
    """
    max_pos_span = np.max([ranges[dim][1] - ranges[dim][0] for dim in
                           dims])
    for k in ranges:
        ranges[k] = list(ranges[k])

    for dim in dims:
        midpoint = 0.5 * (ranges[dim][1] + ranges[dim][0])
        # import pdb; pdb.set_trace()
        ranges[dim][1] = midpoint + 0.5 * max_pos_span
        ranges[dim][0] = midpoint - 0.5 * max_pos_span

LABELS = 'xyzuvw'

if PLOT_CORNER:
    chain_files = [
        '../../results/em_fit/beta_Pictoris_wgs_inv2_5B_res/final_chain.npy',
        '../../results/em_fit/beta_Pictoris_wgs_inv2_5B_tuc-hor_res/final_chain.npy',
    ]
    plot_names = [
        'bpmg_5B_corner.pdf',
        'tuc-hor_5B_corner.pdf',
    ]
    for chain_file, plot_name in zip(chain_files, plot_names):
        axis_labels = [
            'X [pc]',
            'Y [pc]',
            'Z [pc]',
            'U [km/s]',
            'V [km/s]',
            'W [km/s]',
            r'$\sigma_{xyz}$ [pc]',
            r'$\sigma_{uvw}$ [km/s]',
            't [Myr]',
        ]
        print("Plotting {}".format(plot_name))
        chain = np.load(chain_file).reshape(-1,9)
        chain[:,6:8] = np.exp(chain[:,6:8])
        # plt.tick_params(direction='in')
        fig = corner.corner(
            chain,
            labels=axis_labels,
            # reverse=True,
            label_kwargs={'fontsize':'xx-large'},
            max_n_ticks=4,
        )
        print("Applying tick parameters")
        for ax in fig.axes:
            ax.tick_params(direction='in', labelsize='x-large', top=True,
                           right=True)
        print("... saving")
        plt.savefig(plot_name)

if PLOT_BPMG_REAL:
    # PLOTTING ITERATION 6E
    # star_pars_file = '../../data/beta_Pictoris_with_gaia_small_xyzuvw.fits'
    table_file = '../../data/paper1/beta_Pictoris_corrected_everything.fits'
    bpmg_table = Table.read(table_file)
    star_pars, table_ixs = tabletool.build_data_dict_from_table(bpmg_table,
                                                                return_table_ixs=True)
    nstars = len(table_ixs[0])
    fit_name = 'bpmg_and_nearby'
    rdir = '../../results/beta_Pictoris_with_gaia_small_inv2/6/E/final/'

    memb_file = rdir + 'final_membership.npy'
    comp_file = rdir + 'final_comps.npy'

    z = np.load(memb_file)
    comps = SphereComponent.load_raw_components(comp_file)

    # Assign markers based on BANYAN membership
    banyan_markers = np.array(nstars * ['.'])
    banyan_membs = bpmg_table['banyan_assoc'][table_ixs]

    # Assign markers to each star (via `banyan_markers`) whilst concurrently
    # building list of maker style and the corresponding association for legend
    marker_style = []
    marker_label = []
    banyan_memb_set = set(banyan_membs)
    for bassoc in banyan_memb_set:
        if bassoc not in acronyms.keys():
            acronyms[bassoc] = bassoc
    # Do BPMG manually
    banyan_markers[np.where(banyan_membs=='beta Pictoris')] = 'v'
    banyan_memb_set.remove('beta Pictoris')
    marker_label.append(acronyms['beta Pictoris'])
    marker_style.append('v')

    # Do Tucana-Horologium manually
    banyan_markers[np.where(banyan_membs=='Tucana-Horologium')] = '*'
    banyan_memb_set.remove('Tucana-Horologium')
    marker_label.append(acronyms['Tucana-Horologium'])
    marker_style.append('*')

    # Do remainder automatically
    banyan_memb_set.remove('N/A') # remove 'N/A' as they will remain default '.'
    for banyan_assoc, marker in zip(banyan_memb_set, ('s', 'p', 'D', 'X', 'H', 'D')): #''''''^', '<', '>', '8', 's', 'p', 'h', 'H', 'D', 'd', 'P', 'X')):
        banyan_markers[np.where(banyan_membs==banyan_assoc)] = marker
        marker_label.append(acronyms[banyan_assoc])
        marker_style.append(marker)

    # First do all, then just do possible membs of BPMG
    if True:
        # x_nearby_ranges, y_nearby_ranges =\
        #     calcRanges(star_pars, sep_axes=True, scale=False)
        # nearby_star_pars = {}
        # for key in ['xyzuvw', 'xyzuvw_cov']:
        #     nearby_star_pars[key] = np.copy(star_pars[key])
        #
        # # Replace cov matrices with None for bg stars
        # nearby_star_pars['xyzuvw_cov'][
        #     np.where(z.argmax(axis=1)==z.shape[1]-1)
        # ] = None

        # Set to None all covariance matrices not part of BPMG or THOR
        bpmg_ix = 0
        thor_ix = 3
        thor2_ix = 4
        bg_mask = np.where(np.logical_not(
            np.isin(np.argmax(z, axis=1), [bpmg_ix,thor_ix, thor2_ix])
        ))
        nearby_star_pars = {}
        mns = star_pars['means']
        covs = star_pars['covs']
        covs[bg_mask] = None
        # nearby_star_pars['xyzuvw'] = star_pars['xyzuvw']
        # nearby_star_pars['xyzuvw_cov'] = np.copy(star_pars['xyzuvw_cov'])
        # nearby_star_pars['xyzuvw_cov'][bg_mask] = None
        # nearby_star_pars['indices'] = np.array(star_pars['indices'])

        bpmg_xranges, bpmg_yranges = calcRanges(
                {'xyzuvw':star_pars['means']}, sep_axes=True,
        )

        plt.clf()
        fig, axes = plt.subplots(2,2, figsize=(10,10))

        BPMG_COLORS = COLORS[:z.shape[1]]
        BPMG_COLORS[-1] = 'xkcd:grey'
        background_ix = z.shape[1] - 1

        for ax, (dim1, dim2) in zip(axes.flatten(), DEFAULT_DIMS):

            [c.plot(ax=ax, dim1=dim1, dim2=dim2, color=BPMG_COLORS[ix],
                    msize=30, alpha=0.4, marker_alpha=0.)
                for ix, c in enumerate(comps)]

            for ix, (mn, cov) in enumerate(zip(mns, covs)):
                comp_memb_ix = np.argmax(z[ix])

                star_alpha = 0.45 if comp_memb_ix != background_ix else 0.2

                star_zorder = 8 if comp_memb_ix != background_ix else 2
                if (comp_memb_ix == background_ix) and (banyan_markers[ix] == '.'):
                    marker_size = 20
                else:
                    marker_size = 80

                if np.any(np.isnan(cov)):
                    ax.scatter(mn[dim1], mn[dim2], color=BPMG_COLORS[comp_memb_ix],
                               marker=banyan_markers[ix], alpha=star_alpha,
                               s=marker_size)
                else:
                    star_comp = FreeComponent(attributes={'mean':mn,
                                                          'covmatrix':cov,
                                                          'age':0.})
                    star_comp.plot(ax=ax, dim1=dim1, dim2=dim2,
                                   color=BPMG_COLORS[comp_memb_ix],
                                   marker_alpha=star_alpha,
                                   alpha=0.25,
                                   marker=banyan_markers[ix],
                                   msize=70, comp_now=False, comp_then=True,
                                   mzorder=star_zorder,
                                   comp_then_linewidth=0.01)
            ax.set_xlabel('{} {}'.format(labels[dim1], units[dim1]))
            ax.set_ylabel('{} {}'.format(labels[dim2], units[dim2]))

            if int(dim1 / 3) == int(dim2 / 3):
                ax.axis('equal')

            scaleRanges(bpmg_xranges, (0, 1, 2))
            scaleRanges(bpmg_xranges, (3, 4, 5))
            scaleRanges(bpmg_yranges, (0, 1, 2))
            scaleRanges(bpmg_yranges, (3, 4, 5))

            ax.set_xlim(bpmg_xranges[dim1])
            ax.set_ylim(bpmg_yranges[dim2])
            ax.tick_params(direction='in', top=True, right=True)

        # Manually insert marker and color legend, coincidentally the most
        # recent 'ax' is the Z-W plot, as desired
        for label, style in zip(marker_label, marker_style):
            ax.scatter(1e10, 1e10, c='black',
                       marker=style,
                       label=label)
        for ix, color in enumerate(BPMG_COLORS[:-1]):
            label = 'Fitted {}'.format(chr(ord('A') + ix))
            ax.plot(1e10, 1e10, c=color, label=label)

        ax.scatter(1e10, 1e10, c='xkcd:grey', alpha=0.5, label='background')
        ax.legend(loc='best', prop={'size': 9})
        fig.set_tight_layout(tight=True)

        plt.savefig('corrected_bpmg.pdf')

        if False:
        # for dim1, dim2 in DEFAULT_DIMS: #[(0,1), (0,3), (1,4), (2,5)]: #, 'yv', 'zw']:
            # # force the XY plot to have same scales
            # if dim1==0 and dim2==1 and debugging_circles:
            #     temp_range = nearby_range[1]
            #     nearby_range[1] = [-120,80]




            x_nearby_ranges[dim1], y_nearby_ranges[dim2] = fp.plotPane(
                dim1,
                dim2,
                groups=groups,
                star_pars=nearby_star_pars,
                group_now=True,
                membership=z,
                # true_memb=true_memb,
                savefile='redo_{}_{}_{}{}.pdf'.format(fit_name, iteration,
                                                 LABELS[dim1], LABELS[dim2]),
                with_bg=True,
                range_1=x_nearby_ranges[dim1], #range_1,
                range_2=y_nearby_ranges[dim2], #range_2,
                markers=banyan_markers,
                marker_style=marker_style,
                marker_labels=marker_label if dim1 == 2 else None,
                color_labels=COLOR_LABELS[:len(groups)] if
                               dim1 == 2 else None,
                isotropic=(int(dim1/3) == int(dim2/3)),
            )
            # # undo forced change
            # if dim1 == 0 and dim2 == 1 and debugging_circles:
            #     nearby_range[1] = temp_range
            scaleRanges(x_nearby_ranges, (0,1,2))
            scaleRanges(x_nearby_ranges, (3,4,5))
            # scaleRanges(y_nearby_ranges, (0,1,2))
            scaleRanges(y_nearby_ranges, (3,4,5))

    # Only include stars that, if they weren't bg, they'd most likely be BPMG
    if False:
        if iteration == '5B':
            fit_name = 'bpmg_candidates'
            # extract_group_ix = [0,2]
            extract_group_ixs_by_iteration = {
                '5B':[0,3],
                '6C':[0,2],
            }
            extract_group_ix = extract_group_ixs_by_iteration[iteration]
            # bpmg_mask = np.where(z[:,extract_group_ix]>0.1)
            bpmg_star_pars = {}
            # bpmg_mask = np.where(np.isin(np.argmax(z[:,:-1], axis=1), extract_group_ix))# == extract_group_ix)
            bpmg_mask = np.where(np.isin(np.argmax(z, axis=1), extract_group_ix))# == extract_group_ix)
            bg_mask = np.where(np.logical_not(
                np.isin(np.argmax(z, axis=1), extract_group_ix)
            ))
            bpmg_star_pars['xyzuvw'] = star_pars['xyzuvw'] #[bpmg_mask]
            bpmg_star_pars['xyzuvw_cov'] = np.copy(star_pars['xyzuvw_cov']) #[bpmg_mask]
            bpmg_star_pars['xyzuvw_cov'][bg_mask] = None
            bpmg_star_pars['indices'] = np.array(star_pars['indices']) #[bpmg_mask]

            # z = z[bpmg_mask]#, (0,-1),]
            z = z[:,(extract_group_ix+[-1]),]

            # bpmg_range = calcRanges(bpmg_star_pars)
            # import pdb; pdb.set_trace()

            for dim1, dim2 in DEFAULT_DIMS: #[(0,1), (0,3), (1,4)]: #, (2,5)]: #, 'yv', 'zw']:
                # force the XY plot to have same scales
                # if dim1==0 and dim2==1 and debugging_circles:
                #     temp_range = bpmg_range[1]
                #     bpmg_range[1] = [-120,80]
                # import pdb; pdb.set_trace()

                dim1_range, dim2_range = fp.plotPane(
                    dim1,
                    dim2,
                    groups=groups[extract_group_ix],
                    star_pars=bpmg_star_pars,
                    group_now=True,
                    membership=z,
                    savefile='{}_{}_{}{}.pdf'.format(fit_name,
                                                     iteration,
                                                     LABELS[dim1],
                                                     LABELS[dim2]),
                    with_bg=True,
                    # range_1=bpmg_range[dim1],
                    range_1=x_nearby_ranges[dim1],
                    # range_2=bpmg_range[dim2],
                    range_2=y_nearby_ranges[dim2],
                    # residual=True,
                    markers=banyan_markers,
                    marker_style=marker_style,
                    marker_labels=marker_label if dim1==2 else None,
                    color_labels=[r'Fitted $\beta$PMG'] if dim1==2 else None,
                    # isotropic=(int(dim1/3) == int(dim2/3))
                )
                # # undo forced change
                # if dim1 == 0 and dim2 == 1 and debugging_circles:
                #     bpmg_range[1] = temp_range

    # To ensure consistency, we now plot the BANYAN bpmg stars only,
    # and use the ragnes from previous plot
    # 2019-07-13 [TC]: not sure why this is here...
    if False:
        fit_name = 'banyan_bpmg'
        rdir = '../../results/em_fit/beta_Pictoris/'

        memb_file = rdir + 'final_membership.npy'
        groups_file = rdir + 'final_best_groups.npy'
        star_pars_file = '../../data/beta_Pictoris_with_gaia_small_xyzuvw.fits'

        z = np.load(memb_file)
        groups = dt.loadGroups(groups_file)
        star_pars = dt.loadDictFromTable(star_pars_file, 'beta Pictoris')
        nstars = len(star_pars['xyzuvw'])

        # First do all, then just do possible membs of BPMG
        for dim1, dim2 in DEFAULT_DIMS: #[(0,1), (0, 3), (1, 4), (2,5)]: #, (2, 5)]:  # , 'yv', 'zw']:
            # if dim1 == 0 and dim2 == 1 and debugging_circles:
            #     temp_range = bpmg_range[1]
            #     bpmg_range[1] = [-120, 80]
            # import pdb; pdb.set_trace()
            fp.plotPane(
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
                range_1=x_nearby_ranges[dim1],
                range_2=y_nearby_ranges[dim2],
                markers=nstars*['v'],
                marker_labels=[r'BANYAN $\beta$PMG'] if dim1==2 else None,
                color_labels=[r'Chronostar $\beta$PMG'] if dim1==2 else None,
                isotropic=(int(dim1/3) == int(dim2/3)),
            )
            # undo forced change
            # if dim1 == 0 and dim2 == 1 and debugging_circles:
            #     bpmg_range[1] = temp_range

def plot_comps_and_stars(dim1, dim2, star_pars, comps, Component=SphereComponent,
                         star_orbits=False, star_age=0., star_then=False,
                         comp_kwargs={},
                         ):
    if type(star_pars) is str:
        tabletool.build_data_dict_from_table(star_pars)
    if type(comps) is str:
        comps = Component.load_raw_components(comps)




# --------------------------------------------------
# --  PLOTTING FEDERRATH STARS  --------------------
# --------------------------------------------------
# Plots the following:
# - current day measured star positions
# - best fit
# - gaussian fit to true origin
# - calculated stellar traceback orbits
#   (with incorporated measurement uncertainties)
if PLOT_FED_STARS:
    print("Plotting fed stars)")

    # Setting up file names
    synth_fit = 'fed_stars'
    rdir = '../../results/archive/fed_fits/20/gaia/'
    origins_file = rdir + 'origins.npy'
    origin_comp_file = rdir + 'origin_ellip_comp.npy'
    chain_file = rdir + 'final_chain.npy'
    lnprob_file = rdir + 'final_lnprob.npy'
    star_pars_file = rdir + 'xyzuvw_now.fits'
    init_xyzuvw_file = rdir + '../xyzuvw_init_offset.npy'
    perf_xyzuvw_now = rdir + '../perf_xyzuvw.npy'

    # loading in data
    best_comp = SphereComponent.get_best_from_chain(chain_file, lnprob_file)
    origin_comp = EllipComponent.load_raw_components(origin_comp_file)[0]
    init_xyzuvw = np.load(init_xyzuvw_file)
    star_pars = dt.loadXYZUVW(star_pars_file)
    perf_mean_now = np.load(perf_xyzuvw_now)

    original_origin = dt.loadGroups(origins_file)[0]


    # assigning useful shorthands
    mns = star_pars['xyzuvw']
    covs = star_pars['xyzuvw_cov']

    fed_xranges, fed_yranges = calcRanges(
        {'xyzuvw':np.vstack((star_pars['xyzuvw'],init_xyzuvw))},
        sep_axes=True,
    )

    labels='XYZUVW'
    units = 3*['[pc]'] + 3*['[km/s]']
    # import pdb; pdb.set_trace()

    plt.clf()
    fig, axes = plt.subplots(2, 2, figsize=(8,8))

    for ax, (dim1, dim2) in zip(axes.flatten(), DEFAULT_DIMS):
        # Plot the initial distribution in solid red, with orbit
        origin_comp.plot(ax=ax, dim1=dim1, dim2=dim2, color='xkcd:red', msize=30,
                         comp_then=True, comp_now=False, comp_orbit=True,
                         alpha=0.8, orbit_alpha=0.7, comp_then_fill=False,
                         comp_then_linewidth='2')
        # Plot the initial stars
        for init_mean in init_xyzuvw:
            ax.scatter(init_mean[dim1], init_mean[dim2], marker='s', color='xkcd:grey',
                        alpha=0.7)
        # Plot the stars as they are today
        for mean in perf_mean_now:
            ax.scatter(mean[dim1], mean[dim2], marker='*', color='black',
                       s=25,
                       alpha=0.7)
        # Plot the fit to today
        best_comp.plot(ax=ax, dim1=dim1, dim2=dim2, color='xkcd:red', msize=30,
                       alpha=0.7)
        # Plot the measurements of the stars
        for ix, (mn, cov) in enumerate(zip(mns, covs)):
            star_comp = FreeComponent(attributes={'mean':mn,
                                                  'covmatrix':cov,
                                                  'age':-20.})
            star_comp.plot(ax=ax, dim1=dim1, dim2=dim2, color='xkcd:blue',
                           comp_orbit=(ix%3==0), comp_now=False, comp_then=True,
                           msize=50, marker='o', orbit_alpha=0.7,
                           orbit_color='xkcd:grey', orbit_arrow=False)

        ax.set_xlabel('{} {}'.format(labels[dim1], units[dim1]))
        ax.set_ylabel('{} {}'.format(labels[dim2], units[dim2]))

        # Standardise ranges
        if int(dim1/3) == int(dim2/3):
            ax.axis('equal')

        fed_xranges[dim1] = ax.get_xlim()
        fed_yranges[dim2] = ax.get_ylim()

        scaleRanges(fed_xranges, (0, 1, 2))
        scaleRanges(fed_xranges, (3, 4, 5))
        scaleRanges(fed_yranges, (0, 1, 2))
        scaleRanges(fed_yranges, (3, 4, 5))

        ax.set_xlim(fed_xranges[dim1])
        ax.set_ylim(fed_yranges[dim2])
        ax.tick_params(direction='in', top=True, right=True)

        # plt.savefig('{}_both_{}{}.pdf'.format(synth_fit,
        #                                       labels[dim1],
        #                                       labels[dim2]))
    fig.set_tight_layout(tight=True)
    fig.savefig('{}_all.pdf'.format(synth_fit))


    # for dim1, dim2 in DEFAULT_DIMS: #[(0,1), (0,3), (1,4), (2,5)]:
    #     # plt.clf()
    #     fed_xranges[dim1], fed_yranges[dim2] = fp.plotPane(
    #         dim1,
    #         dim2,
    #         groups=comps,
    #         star_pars=star_pars_file,
    #         origin_star_pars={'xyzuvw':init_xyzuvw},
    #         group_then=True,
    #         group_now=True,
    #         star_orbits=True,
    #         savefile='{}_both_{}{}.pdf'.format(synth_fit,
    #                                            LABELS[dim1],
    #                                            LABELS[dim2]),
    #         marker_legend={'current-day':'.', 'origin':'s'} if dim1==2 else None,
    #         color_legend={'current-day':'xkcd:red', 'origin':'xkcd:orange'} if dim1==2 else None,
    #         star_pars_label='current-day',
    #         origin_star_pars_label='origin',
    #         isotropic=(int(dim1/3) == int(dim2/3)),
    #         range_1=fed_xranges[dim1],
    #         range_2=fed_yranges[dim2],
    #     )
    #     scaleRanges(fed_xranges, (0, 1, 2))
    #     scaleRanges(fed_xranges, (3, 4, 5))
    #     # scaleRanges(fed_yranges, (0, 1, 2))
    #     # scaleRanges(fed_yranges, (3, 4, 5))
    #     # scaleRanges(fed_xranges, (0,1,2))
    #     # scaleRanges(fed_xranges, (3,4,5))
    #

# plotting Multi-component synth fits
if PLOT_MUTLI_SYNTH:
    print("Plotting synth plots")
    synth_fits = [
        # 'synth_bpmg',
        'four_assocs',
        'assoc_in_field',
        'same_centroid',
        # 'synth_bpmg2',
    ]

    rdir_suffix = {
        # 'synth_bpmg':'',
        'four_assocs':'_res',
        'assoc_in_field':'_res',
        'same_centroid':'_res',
        'synth_bpmg2':'_res',
    }

    planes = {
        # 'synth_bpmg':['xu', 'zw'], #['xu', 'zw', 'xy']#, 'yz'],
        'four_assocs':['xu', 'zw'], #['xy', 'yv'],
        'assoc_in_field':['xu', 'zw'], #['uv', 'xu'],
        'same_centroid':['xu', 'zw'], #['xu', 'yv'],
        'synth_bpmg2':['xu', 'zw'], #['xu', 'zw', 'xy']#, 'yz'],
    }

    with_bg = {
        # 'synth_bpmg':True,
        'four_assocs':False,
        'assoc_in_field':False,
        'same_centroid':False,
        'synth_bpmg2':True,
    }

    ordering = {
        # 'synth_bpmg':[1, 0],
        'assoc_in_field':[1, 0],
        'four_assocs':[3, 2, 0, 1],
        'same_centroid':[1, 0],
        'synth_bpmg2':[1, 0],
    }

    legend_proj = {
        # 'synth_bpmg':(0,3),
        'assoc_in_field':(2,5),
        'four_assocs':(2,5),
        'same_centroid':(2,5),
        'synth_bpmg2':(0,3),
    }

    MARKER_LABELS = np.array(['True {}'.format(ch) for ch in 'ABCD'])

    for synth_fit in synth_fits:
        print(" - plotting {}".format(synth_fit))
        rdir = '../../results/em_fit/{}{}/'.format(synth_fit,
                                                   rdir_suffix[synth_fit])
        groups_file = rdir + 'final_best_groups.npy'
        # star_pars_file = rdir + '{}_xyzuvw.fits'.format(synth_fit)
        groups = dt.loadGroups(groups_file)
        star_pars_file = '../../data/{}_xyzuvw.fits'.format(synth_fit)
        memb_file = rdir + 'final_membership.npy'
        origins_file = rdir + 'synth_data/origins.npy'
        true_memb = dt.getZfromOrigins(origins_file, star_pars_file)
        ranges = calcRanges(dt.loadXYZUVW(star_pars_file))
        xaxis_ranges, yaxis_ranges = calcRanges(dt.loadXYZUVW(star_pars_file),
                                                 sep_axes=True, scale=True)
        # yaxis_ranges = {}
        # for key in ranges.keys():
        #     xaxis_ranges[key] = ranges[key][:]
        #     yaxis_ranges[key] = ranges[key][:]


        for dim1, dim2 in DEFAULT_DIMS: #planes[synth_fit]:
            print("   - {} and {}".format(dim1, dim2))
            # import pdb; pdb.set_trace()
            xaxis_ranges[dim1], yaxis_ranges[dim2] = fp.plotPaneWithHists(
                dim1,
                dim2,
                groups=groups_file,
                star_pars=star_pars_file,
                group_now=True,
                membership=memb_file,
                true_memb=true_memb,
                savefile='{}_{}{}.pdf'.format(synth_fit,
                                              LABELS[dim1],
                                              LABELS[dim2]),
                with_bg=with_bg[synth_fit],
                group_bg=(synth_fit == 'assoc_in_field'),
                isotropic=(int(dim1/3) == int(dim2/3)),
                range_1=xaxis_ranges[dim1],
                range_2=yaxis_ranges[dim2],
                color_labels=COLOR_LABELS[:len(groups)]
                        if (dim1, dim2) == legend_proj[synth_fit]
                        else None,
                marker_labels=MARKER_LABELS[:len(groups)] #[ordering[synth_fit]]
                        if (dim1, dim2) == legend_proj[synth_fit]
                        else None,
                ordering=ordering[synth_fit],
                marker_order=ordering[synth_fit],
                no_bg_covs=with_bg[synth_fit],
            )
            # import pdb; pdb.set_trace()
            scaleRanges(xaxis_ranges, (0, 1, 2))
            scaleRanges(xaxis_ranges, (3, 4, 5))
            # scaleRanges(yaxis_ranges, (0, 1, 2))
            # scaleRanges(yaxis_ranges, (3, 4, 5))


# plotting Multi-component synth fits
if PLOT_SYNTH_BPMG2:
    print("Plotting synth plots")
    synth_fits = [
        'synth_bpmg2',
    ]

    rdir_suffix = {
        'synth_bpmg2':'_res',
    }

    planes = {
        'synth_bpmg2':['xu', 'zw'], #['xu', 'zw', 'xy']#, 'yz'],
    }

    with_bg = {
        'synth_bpmg2':True,
    }

    ordering = {
        'synth_bpmg2':[1, 0],
    }

    legend_proj = {
        'synth_bpmg2':(0,3),
    }

    MARKER_LABELS = np.array(['True {}'.format(ch) for ch in 'ABCD'])

    for synth_fit in synth_fits[-1:]:
        print(" - plotting {}".format(synth_fit))
        rdir = '../../results/em_fit/{}{}/'.format(synth_fit,
                                                   rdir_suffix[synth_fit])
        groups_file = rdir + 'final_best_groups.npy'
        # star_pars_file = rdir + '{}_xyzuvw.fits'.format(synth_fit)
        groups = dt.loadGroups(groups_file)
        star_pars_file = '../../data/{}_xyzuvw.fits'.format(synth_fit)
        memb_file = rdir + 'final_membership.npy'
        origins_file = rdir + 'synth_data/origins.npy'
        true_memb = dt.getZfromOrigins(origins_file, star_pars_file)
        ranges = calcRanges(dt.loadXYZUVW(star_pars_file))
        xaxis_ranges, yaxis_ranges = calcRanges(dt.loadXYZUVW(star_pars_file),
                                                 sep_axes=True, scale=True)
        # yaxis_ranges = {}
        # for key in ranges.keys():
        #     xaxis_ranges[key] = ranges[key][:]
        #     yaxis_ranges[key] = ranges[key][:]


        for dim1, dim2 in DEFAULT_DIMS: #planes[synth_fit]:
            print("   - {} and {}".format(dim1, dim2))
            # import pdb; pdb.set_trace()
            xaxis_ranges[dim1], yaxis_ranges[dim2] = fp.plotPane(
                dim1,
                dim2,
                groups=groups_file,
                star_pars=star_pars_file,
                group_now=True,
                membership=memb_file,
                true_memb=true_memb,
                savefile='{}_{}{}.pdf'.format(synth_fit,
                                              LABELS[dim1],
                                              LABELS[dim2]),
                with_bg=with_bg[synth_fit],
                group_bg=(synth_fit == 'assoc_in_field'),
                isotropic=(int(dim1/3) == int(dim2/3)),
                range_1=xaxis_ranges[dim1],
                range_2=yaxis_ranges[dim2],
                color_labels=COLOR_LABELS[:len(groups)]
                        if (dim1, dim2) == legend_proj[synth_fit]
                        else None,
                marker_labels=MARKER_LABELS[:len(groups)] #[ordering[synth_fit]]
                        if (dim1, dim2) == legend_proj[synth_fit]
                        else None,
                ordering=ordering[synth_fit],
                # marker_order=ordering[synth_fit],
                no_bg_covs=with_bg[synth_fit],
            )
            # import pdb; pdb.set_trace()
            scaleRanges(xaxis_ranges, (0, 1, 2))
            scaleRanges(xaxis_ranges, (3, 4, 5))
            # scaleRanges(yaxis_ranges, (0, 1, 2))
            # scaleRanges(yaxis_ranges, (3, 4, 5))


if PLOT_FAILURE:
    synth_fit='failure_mode'
    labels = ['a', 'b']
    groups = []
    for label in labels:
        rdir = '../../results/synth_fit/30_2_1_25_{}_double/'.format(label)
            # rdir = '../../results/new_fed_stars_20/gaia/'
        # origins_file = rdir + 'origins.npy'
        chain_file = rdir + 'final_chain.npy'
        lnprob_file = rdir + 'final_lnprob.npy'
        # init_xyzuvw_file = '../../data/sink_init_xyzuvw.npy'
        # init_xyzuvw_file = rdir + '../xyzuvw_init_offset.npy'
        # perf_xyzuvw_file = rdir + '../perf_xyzuvw.npy'
        # star_pars_file = '../../data/fed_stars_20_xyzuvw.fits'

        chain = np.load(chain_file).reshape(-1,9)
        lnprobs = np.load(lnprob_file)
        # best_fit_pars = np.load(chain_file)[np.unravel_index(np.argmax(lnprobs), lnprobs.shape)]
        best_fit_pars = chain[np.argmax(lnprobs)]
        groups.append(
            chronostar.component.Component(best_fit_pars, internal=True))
        # origins = dt.loadGroups(origins_file)
        # raw_init_xyzuvw = np.load(init_xyzuvw_file)
        # perf_xyzuvw = np.load(perf_xyzuvw_file)
        # init_xyzuvw = torb.traceManyOrbitXYZUVW(perf_xyzuvw, -origins[0].age,
        #                                         single_age=True)
        # init_xyzuvw = np.load(init_xyzuvw_file)


    # this luckiliy picks out sample 'b' which is what we want.
    star_pars_file = rdir + 'xyzuvw_now.fits'

    for dim1, dim2 in DEFAULT_DIMS: #['xy', 'xu', 'yv', 'zw', 'uv', 'uw']:
        fp.plotPane(
            dim1,
            dim2,
            groups=groups[::-1], # reverse groups so failure is coloured
            star_pars=star_pars_file,
            group_now=True,
            group_then=True,
            star_orbits=True,
            group_orbit=True,
            membership=None,
            true_memb=None,
            savefile='{}_{}{}.pdf'.format(synth_fit,
                                          LABELS[dim1],
                                          LABELS[dim2]),
            isotropic=(int(dim1/3) == int(dim2/3)),
        )
