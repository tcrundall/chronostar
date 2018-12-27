from __future__ import print_function, division

"""
Gather the parameter fits for a bunch of synth fits for large scale
statistical analysis of single component fitting performance

a 'med_and_span.npy' file stores a [9,3] np.float array
The first dimension corresponds to values
[x, y, z, u, v, w, lndx, lndv, age]
the second dimension corresponds to percentile values:
[50th, 84th, 16th] 


Properties I want:
- bin sizes the same
    (ideally construct histograms from same bin template)
- xaxis broken [SOLVED]
- plot bars with hatching

"""

from itertools import product
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '..')

def gauss(x, mu, sig):
    """
    Evaluates a 1D gaussian at `x` given mean `mu` and std `sig`
    """
    return 1./(sig*np.sqrt(2*np.pi)) * np.exp(-(x - mu)**2 / (2.*sig**2))

def insertSpanIntoDict(dict, key_list, med_and_span):
    """
    Inserts array `med_and_span` into the appropriate sub-dictionary, building
    dictionary paths as required using the values in `scenario` and `prec`
    as keys.
    """
    next_key = key_list.pop(0)
    # if no more keys in key_list, then insert med_and_span
    if key_list == []:
        dict[next_key] = med_and_span
        return

    # if keys still remain yet current dict doesn't yet have `next_key`, insert
    # empty dictionary
    if next_key not in dict.keys():
        dict[next_key] = {}

    insertSpanIntoDict(dict[next_key], key_list, med_and_span)


def getScenarioFromIndex(index):
    """For investigative purposes: given an index for a realisation,
    prints the parameter values"""
    prec_ix = int( index % len(precs) )
    label_ix = int( (index / len(precs)) % len(labels) )
    nstar_ix = int( (index / len(precs) / len(labels)) % len(nstars) )
    dv_ix = int( (index / len(precs) / len(labels) / len(nstars)) % len(dvs) )
    dx_ix = int( (index / len(precs) / len(labels) / len(nstars) / len(dvs)) % len(dxs) )
    print(dxs[dx_ix], dvs[dv_ix], nstars[nstar_ix], labels[label_ix], precs[prec_ix])

def breakAxes(ax1, ax2, ymin=None, ymax=None): #, horizontal=True):
    """
    Given two axes objects, paint as if a single plot with broken axes in
    x (Currently hardcoded for only x direction)

    Can optionally provide the ylimits.
    """

    # line up y axes across two axes objects
    if ymin is None:
        ymin = np.min((ax1.get_ylim()[0], ax2.get_ylim()[0]))
    if ymax is None:
        ymax = np.max((ax1.get_ylim()[1], ax2.get_ylim()[1]))
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    # Remove unneeded spines
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax1.tick_params(labelright='off')  # don't put tick labels at the right
    ax2.tick_params(labelleft='off')
    # ax2.tick_params(labelright='off')  # don't put tick labels at the right
    ax2.yaxis.tick_right()

    # and break marks
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the right axes
    ax2.plot((-d, +d), (-d, +d), **kwargs)  # bottom-left diagonal
    ax2.plot((- d, + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

def plotAgeHistsByPrec(ax1, ax2, age, precs, bad_hists, bad_edges, hists, edges,
                       weight=1.):
    alpha=0.7
    ax2.tick_params(direction='in', top=True)#, right=True)
    for prec in precs:
        width = np.diff(edges[age][prec])
        # center = (edges[age][prec][:-1] + edges[age][prec][1:]) / 2
        ax2.bar(
            edges[age][prec][:-1],
            hists[age][prec]*weight,
            width=width,
            align='edge',
            alpha=alpha,
            hatch=patterns[precs.index(prec)],
            color='none',
            edgecolor=list(plt.rcParams['axes.prop_cycle'])[precs.index(prec)]['color'],# plt.rcParams['axes.prop_cycle'][precs.index(prec)],
            label=prec_labels[prec],
        )

    # scale of plot is different if not using broken axis
    if bad_hists is not None:
        xpos = 0.96
    else:
        xpos = 0.98
    ax2.text(xpos, 0.92, r"$t_{{true}} =${:2} Myr".format(age),
                    fontsize=10,
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax2.transAxes
                    )

    if bad_hists is not None:
        ax1.tick_params(direction='in', top=True, left=True)
        for prec in precs:
            ax1.bar(
                bad_edges[age][prec][:-1],
                bad_hists[age][prec],
                width=width,
                alpha=alpha,
                align='edge',
                hatch=patterns[precs.index(prec)],
                color='none',
                edgecolor=list(plt.rcParams['axes.prop_cycle'])[precs.index(prec)]['color'], #plt.rcParams['axes.prop_cycle'][precs.index(prec)],
                label=prec_labels[prec],
            )
        breakAxes(ax1, ax2, ymin=0., ymax=1.25)
    else:
        ax2.tick_params(right=True)
        ax2.set_ylim(0,1.25)

def calcNumpyHists(ages, precs, data, good_mask, bad_mask, info):
    """
    Calculates normalised, standardised (re: bin placement) histograms
    of data
    """
    max_raw_res = max(
        [max(abs(raw_resids_by_age[age][vanilla_mask[age]])) for age in ages])
    raw_span = [-max_raw_res, max_raw_res]

    # get histograms through numpy beforehand (for scaling reasons)
    hists = {}
    edges = {}
    max_bin_height = {}
    bad_hists = {}
    bad_edges = {}
    for age in ages:
        hists[age] = {}
        edges[age] = {}
        bad_hists[age] = {}
        bad_edges[age] = {}
        for prec in precs:
            hists[age][prec], edges[age][prec] = np.histogram(
                data[info][age][prec][good_mask[age][prec]],
                bins=19,
                range=raw_span,
            )
            if bad_mask is not None:
                bad_hists[age][prec], bad_edges[age][prec] = np.histogram(
                    data[info][age][prec][bad_mask[age][prec]],
                    bins=19,
                    range=np.array(raw_span) - 19
                )
        max_bin_height[age] = np.max([hists[age][prec] for prec in precs])
        for prec in precs:
            hists[age][prec] = hists[age][prec] / float(max_bin_height[age])
            if bad_mask is not None:
                bad_hists[age][prec] = bad_hists[age][prec] / float(max_bin_height[age])
    if bad_mask is None:
        return hists, edges
    else:
        return hists, edges, bad_hists, bad_edges


def buildMasterDict(ages, dxs, dvs, nstars, labels, precs):
    """
    Builds a dictionary of all simulations and their results

    Dictionary has the form:
        master_dict[ages][dxs][dvs][nstars][labels][precs]
            [[dx, dv, age], (raw)
             [dx, dv, age]] (normed)
    """
    scenarios = product(ages, dxs, dvs, nstars, labels)
    master_dict = {}
    for scenario in scenarios:
        for prec in precs:
            sdir = rdir + '{}_{}_{}_{}_{}/'.format(*scenario) + '{}/'.format(prec)
            med_and_spans = np.load(sdir + 'med_and_span.npy')
            # take exponent of lndx and lndv
            med_and_spans[6:8] = np.exp(med_and_spans[6:8])
            med_and_errs = np.zeros((med_and_spans.shape[0], 2))
            med_and_errs[:,0] = med_and_spans[:,0]
            med_and_errs[:,1] = 0.5*(med_and_spans[:,1] - med_and_spans[:,2])
            # import pdb; pdb.set_trace()
            raw_resids = med_and_spans[6:9,0] - \
                         np.array((scenario[1], scenario[2], scenario[0]))
            norm_resids = raw_resids / med_and_errs[6:9,1]
            resids = np.vstack((raw_resids, norm_resids))
            insertSpanIntoDict(master_dict, list(scenario) + [prec], resids)
    return master_dict

def buildSpecialDicts(master_dict, ages, dxs, dvs, nstars, labels, precs,
                      normed=False):
    resids_by_age = {}
    for age in ages:
        resids_by_age[age] = np.array(
            [master_dict[age][dx][dv][nstar][label][prec][int(normed), -1] for dx in dxs for dv
             in dvs for nstar in nstars for label in labels for prec in precs]
        )
    resids_by_age_prec = {}
    for age in ages:
        resids_by_age_prec[age] = {}
        for prec in precs:
            resids_by_age_prec[age][prec] = np.array(
                [master_dict[age][dx][dv][nstar][label][prec][int(normed), -1] for dx in dxs
                 for dv in dvs for nstar in nstars for label in labels]
            )
    return resids_by_age, resids_by_age_prec

def getAreaUnderCurve(xs, ys):
    """Calculate the area under a curve using simple right hand rectangle
    method"""
    dx = xs[1] - xs[0]
    area = np.sum(ys) * dx
    return area


ages = [5,15,30,50,100]#,200]
dxs = [1, 2, 5]
dvs = [1, 2]
nstars = [25, 50, 100]
labels = 'abcd'
precs = ['half', 'gaia', 'double']#, 'quint']

if __name__ == '__main__':
    rdir = '../results/synth_fit/med_2paper1_runs/'

    master_dict = buildMasterDict(ages, dxs, dvs, nstars, labels, precs)
    raw_resids_by_age, raw_resids_by_age_prec = buildSpecialDicts(
        master_dict, ages, dxs, dvs, nstars, labels, precs, normed=False,
    )
    norm_resids_by_age, norm_resids_by_age_prec = buildSpecialDicts(
        master_dict, ages, dxs, dvs, nstars, labels, precs, normed=True,
    )

    xlabels = {
        'raw':'$t_{\\rm fitted} - t_{\\rm true}$',
        'norm':'Normalised residual'
    }
    #data = {'raw':raw_resids_by_age, 'norm':norm_resids_by_age}
    data = {'raw':raw_resids_by_age_prec, 'norm':norm_resids_by_age_prec}
    vanilla_data = {'raw':raw_resids_by_age, 'norm':norm_resids_by_age}

    vanilla_mask = {}
    bad_vanilla_mask = {}
    for age in ages:
        vanilla_mask[age] = np.where(abs(vanilla_data['raw'][age]) < 15)[0]
        bad_vanilla_mask[age] = np.where(abs(vanilla_data['raw'][age]) >= 15)[0]
    bad_data_mask = {}
    data_mask = {}
    for age in ages:
        data_mask[age] = {}
        bad_data_mask[age] = {}
        for prec in precs:
            data_mask[age][prec] = np.where(abs(data['raw'][age][prec]) < 15)[0]
            bad_data_mask[age][prec] = np.where(abs(data['raw'][age][prec]) >= 15)[0]


    max_raw_res = max([max(abs(raw_resids_by_age[age][vanilla_mask[age]])) for age in ages])
    max_norm_res = max([max(abs(norm_resids_by_age[age][vanilla_mask[age]])) for age in ages])
    raw_span = [-max_raw_res, max_raw_res]
    norm_span = [-max_norm_res, max_norm_res]
    spans = {'raw':raw_span, 'norm':norm_span}

    patterns = ('o', '/', '\\', '-', '+', 'x', '\\', '*', 'o', 'O', '.')

    prec_labels = {
        'half':r'$\eta = 0.5$',
        'gaia':r'$\eta = 1.0$',
        'double':r'$\eta = 2.0$',
    }

    # Plot hists of raw offset with broken axis
    for info in ['raw']: #'['raw', 'norm']:
        f, axes = plt.subplots(
            len(ages),
            2,
            # sharex=True,
            # gridspec_kw={'wspace':0,'hspace':0},
            gridspec_kw={'hspace':0},
            figsize=(5,len(ages))
        )
        axes[-1,1].set_xlabel(xlabels[info])

        # Construct normalised, standard (e.g. bin widths etc) histograms
        hists, edges, bad_hists, bad_edges =\
            calcNumpyHists(ages, precs, data, data_mask, bad_data_mask, info=info)

        # Plot each histogram
        for i, age in enumerate(ages):
            plotAgeHistsByPrec(axes[i,0], axes[i,1], age, precs, bad_hists, bad_edges,
                                   hists, edges)

        # Set primary axes labels and save
        axes[0,0].legend(loc=2, fontsize='small')
        axes[2,0].set_ylabel('Relative frequency [arbitrary units]')
        f.set_tight_layout(tight=True)
        f.savefig('temp_plots/multi_all_{}_step.pdf'.format(info))

    # plot normalised histograms, excluding the failed fits (bad data)
    f, axes = plt.subplots(
        len(ages),
        1,
        gridspec_kw={'hspace': 0},
        figsize=(5, len(ages))
    )
    axes[-1].set_xlabel(xlabels[info])
    hists, edges = \
        calcNumpyHists(ages, precs, data, data_mask, bad_mask=None, info='norm')

    # generate pts for a unit variance Gaussian with a max value of 1 at peak
    xs = np.linspace(np.min(edges[5]['half']), np.max(edges[5]['half']),
                     100)
    ys = gauss(xs, mu=0., sig=1.)
    gauss_area = getAreaUnderCurve(xs, ys)
    ref_area_under_hist = getAreaUnderCurve(edges[5]['half'][:-1],
                                            hists[5]['half'])
    scaled_ys = ys[:]*ref_area_under_hist/gauss_area
    for i, age in enumerate(ages):
        area_under_hist = getAreaUnderCurve(edges[age]['half'][:-1],
                                            hists[age]['half'])
        axes[i].plot(xs, scaled_ys, c='xkcd:black', ls='--')
        plotAgeHistsByPrec(None, axes[i], age, precs, bad_hists=None,
                           bad_edges=None,
                           hists=hists,
                           edges=edges,
                           weight=ref_area_under_hist/area_under_hist,
                           )

    # Set primary axes labels and save
    axes[0].legend(loc=2, fontsize='small')
    axes[2].set_ylabel('Relative frequency [arbitrary units]')
    f.set_tight_layout(tight=True)
    f.savefig('temp_plots/multi_all_{}_step.pdf'.format('norm'))

    if False:
        print("Removed runs:")
        for age in ages:
            print("{:02}: {}".format(age, len(bad_vanilla_mask[age])))

        print("With mean offsets:")
        for age in ages:
            mean_offset = np.mean(vanilla_data['raw'][age][bad_vanilla_mask[age]])
            print("{:3}: {}".format(age, mean_offset))

        print("Specifically:")
        for age in ages:
            print("--- {:3} ---".format(age))
            for ix in bad_vanilla_mask[age]:
                getScenarioFromIndex(ix)

        # all_raw = np.array(raw_resids_by_age.values())
        all_raw = np.hstack([raw_resids_by_age[age][vanilla_mask[age]] for age in ages])
        nfits = len(all_raw.flatten())
        raw_thresh = 0.5
        ngood_raw = np.where((all_raw > -raw_thresh) & (all_raw < raw_thresh))[0].shape[0]
        perc_good_raw = ngood_raw * 100./ nfits
        print("Percentage fits within {} Myr: {:.2f}%".format(raw_thresh, perc_good_raw))

        # all_norm = np.array(norm_resids_by_age.values())
        all_norm = np.hstack([norm_resids_by_age[age][vanilla_mask[age]] for age in ages])
        norm_med = np.median(all_norm)
        #ngood_norm_twosig = np.where((all_norm-norm_med > -2) & (all_norm-norm_med < 2))[0].shape[0]
        ngood_norm_twosig = np.where((all_norm > -2) & (all_norm < 2))[0].shape[0]
        #ngood_norm_threesig = np.where((all_norm-norm_med > -3) & (all_norm-norm_med < 3))[0].shape[0]
        ngood_norm_threesig = np.where((all_norm > -3) & (all_norm < 3))[0].shape[0]
        perc_good_norm_twosig = ngood_norm_twosig * 100. / nfits
        perc_good_norm_threesig = ngood_norm_threesig * 100. / nfits

        print("Percentage fits within 2 sigma: {:.2f}%".format(perc_good_norm_twosig))
        print("Percentage fits within 3 sigma: {:.2f}%".format(perc_good_norm_threesig))
