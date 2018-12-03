from __future__ import print_function, division

"""
Gather the parameter fits for a bunch of synth fits for large scale
statistical analysis of single component fitting performance

a 'med_and_span.npy' file stores a [9,3] np.float array
The first dimension corresponds to values
[x, y, z, u, v, w, lndx, lndv, age]
the second dimension corresponds to percentile values:
[50th, 84th, 16th] 
"""

from itertools import product
import numpy as np
import matplotlib.pyplot as plt

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
    prec_ix = int( index % len(precs) )
    label_ix = int( (index / len(precs)) % len(labels) )
    nstar_ix = int( (index / len(precs) / len(labels)) % len(nstars) )
    dv_ix = int( (index / len(precs) / len(labels) / len(nstars)) % len(dvs) )
    dx_ix = int( (index / len(precs) / len(labels) / len(nstars) / len(dvs)) % len(dxs) )
    print(dxs[dx_ix], dvs[dv_ix], nstars[nstar_ix], labels[label_ix], precs[prec_ix])



rdir = '../results/synth_fit/med_2paper1_runs/'

ages = [5,15,30,50,100]#,200]
dxs = [1, 2, 5]
dvs = [1, 2]
nstars = [25, 50, 100]
labels = 'abcd'
precs = ['half', 'gaia', 'double']#, 'quint']

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
        raw_resids = med_and_spans[6:9,0] -\
                     np.array((scenario[1], scenario[2], scenario[0]))
        norm_resids = raw_resids / med_and_errs[6:9,1]
        resids = np.vstack((raw_resids, norm_resids))
        insertSpanIntoDict(master_dict, list(scenario) + [prec], resids)

raw_resids_by_age = {}
for age in ages:
    raw_resids_by_age[age] = np.array(
        [master_dict[age][dx][dv][nstar][label][prec][0, -1] for dx in dxs for dv
         in dvs for nstar in nstars for label in labels for prec in precs]
    )
raw_resids_by_age_prec = {}
for age in ages:
    raw_resids_by_age_prec[age] = {}
    for prec in precs:
        raw_resids_by_age_prec[age][prec] = np.array(
            [master_dict[age][dx][dv][nstar][label][prec][0, -1] for dx in dxs
             for dv in dvs for nstar in nstars for label in labels]
        )

norm_resids_by_age = {}
for age in ages:
    norm_resids_by_age[age] = np.array(
        [master_dict[age][dx][dv][nstar][label][prec][1, -1] for dx in dxs for dv
         in dvs for nstar in nstars for label in labels for prec in precs]
    )
norm_resids_by_age_prec = {}
for age in ages:
    norm_resids_by_age_prec[age] = {}
    for prec in precs:
        norm_resids_by_age_prec[age][prec] = np.array(
            [master_dict[age][dx][dv][nstar][label][prec][1, -1] for dx in dxs
             for dv in dvs for nstar in nstars for label in labels]
        )



xlabels = {'raw':'Age offset [Myr]', 'norm':'Normalised residual'}
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

# max_raw_res = max([max(abs(val)) for val in raw_resids_by_age.values()])
max_raw_res = max([max(abs(raw_resids_by_age[age][vanilla_mask[age]])) for age in ages])
# max_raw_res = max([max(abs(raw_resids_by_age[age][vanilla_mask[age]]/float(age))) for age in ages])
# max_norm_res = max([max(abs(val)) for val in norm_resids_by_age.values()])
max_norm_res = max([max(abs(norm_resids_by_age[age][vanilla_mask[age]])) for age in ages])
raw_span = [-max_raw_res, max_raw_res]
norm_span = [-max_norm_res, max_norm_res]
spans = {'raw':raw_span, 'norm':norm_span}

patterns = ('o', '/', '\\', '-', '+', 'x', '\\', '*', 'o', 'O', '.')
# vanilla histograms
# factor = {
#     'norm':{},
#     'raw':{}
# }
# for age in ages:
#     factor['norm'][age] = 1.
#     factor['raw'][age] = age

for info in ['raw', 'norm']:
    f, axes = plt.subplots(
        len(ages),
        1,
        sharex=True,
        gridspec_kw={'wspace':0,'hspace':0},
        figsize=(5,len(ages))
    )
    axes[-1].set_xlabel(xlabels[info])
    for i in range(len(ages)):
        axes[i].tick_params(direction='in', top=True, right=True)
        axes[i].hist(
            vanilla_data[info][ages[i]][vanilla_mask[ages[i]]],# / factor[info][age],
            bins=19,
            #bins=bins[i],
            #orientation='horizontal',
            range=spans[info],
            # hatch=patterns[patterns[precs.index(prec)]],
            # alpha=0.5,
            # histtype='step',
        )
        axes[i].text(0.85, 0.80, r"$t_{{true}} =${:2} Myr".format(ages[i]),
                     fontsize=12,
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=axes[i].transAxes
        )
        # axes[i].set_title(ages[i])
        # axes[i].legend(loc='best')
    x_upper = max([ax.get_xlim()[1] for ax in axes])
    x_lower = min([ax.get_xlim()[0] for ax in axes])
    x_max = max(x_upper, abs(x_lower))
    for ax in axes:
        ax.set_xlim(-x_max, x_max)
    axes[1].set_ylabel("Number of runs" + 15*" ")
    axes[0].legend(loc=2)
    f.set_tight_layout(tight=True)
    f.savefig('temp_plots/multi_{}_hist.pdf'.format(info))

prec_labels = {
    'half':'0.5*Gaia error',
    'gaia':'1.0*Gaia error',
    'double':'2.0*Gaia error',
}
# break down hists by measurement error
for info in ['raw', 'norm']:
    f, axes = plt.subplots(
        len(ages),
        1,
        sharex=True,
        gridspec_kw={'wspace':0,'hspace':0},
        figsize=(5,len(ages))
    )
    axes[-1].set_xlabel(xlabels[info])
    for i in range(len(ages)):
        axes[i].tick_params(direction='in', top=True, right=True)
        for prec in precs:
            axes[i].hist(
                data[info][ages[i]][prec][data_mask[ages[i]][prec]],# / factor[info][age],
                bins=19,
                #bins=bins[i],
                #orientation='horizontal',
                range=spans[info],
                histtype='step',
                label=prec_labels[prec],
                hatch=patterns[precs.index(prec)],
                alpha=0.5,
            )
        axes[i].text(0.85, 0.80, r"$t_{{true}} =${:2} Myr".format(ages[i]),
                     fontsize=12,
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=axes[i].transAxes
        )
        # axes[i].set_title(ages[i])
        # axes[i].legend(loc='best')
    x_upper = max([ax.get_xlim()[1] for ax in axes])
    x_lower = min([ax.get_xlim()[0] for ax in axes])
    x_max = max(x_upper, abs(x_lower))
    for ax in axes:
        ax.set_xlim(-x_max, x_max)
    axes[1].set_ylabel("Number of runs" + 15*" ")
    axes[0].legend(loc=2, fontsize='small')
    f.set_tight_layout(tight=True)
    f.savefig('temp_plots/multi_{}_step.pdf'.format(info))

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

# for age in ages:
#     plt.clf()
#     plt.hist(raw_resids_by_age[age])
#     plt.title(age)
#     plt.xlabel('Age offsets [Myr]')
#     plt.savefig('temp_plots/multi_raw_hist_{}.pdf'.format(age))
#
# for age in ages:
#     plt.clf()
#     plt.hist(norm_resids_by_age[age])
#     plt.title(age)
#     plt.xlabel(r'Normalised residuals offsets [$\sigma$]')
#     plt.savefig('temp_plots/multi_norm_hist_{}.pdf'.format(age))
