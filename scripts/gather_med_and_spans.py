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


rdir = '../results/synth_fit/med_paper1_runs/'

ages = [5,15,30,50]
dxs = [1, 2]
dvs = [1, 2]
nstars = [25, 50, 100]
labels = 'abcd'
precs = ['half', 'gaia', 'double']

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

norm_resids_by_age = {}
for age in ages:
    norm_resids_by_age[age] = np.array(
        [master_dict[age][dx][dv][nstar][label][prec][1, -1] for dx in dxs for dv
         in dvs for nstar in nstars for label in labels for prec in precs]
    )

max_raw_res = max([max(abs(val)) for val in raw_resids_by_age.values()])
max_norm_res = max([max(abs(val)) for val in norm_resids_by_age.values()])
raw_span = [-max_raw_res, max_raw_res]
norm_span = [-max_norm_res, max_norm_res]


spans = {'raw':raw_span, 'norm':norm_span}
xlabels = {'raw':'Age offset [Myr]', 'norm':'Normalised residual'}
data = {'raw':raw_resids_by_age, 'norm':norm_resids_by_age}

for info in ['raw', 'norm']:
    f, axes = plt.subplots(
        4,
        1,
        sharex=True,
        gridspec_kw={'wspace':0,'hspace':0},
        figsize=(5,10)
    )
    axes[3].set_xlabel(xlabels[info])
    for i in range(len(ages)):
        axes[i].tick_params(direction='in', top=True, right=True)
        axes[i].hist(
            data[info][ages[i]],
            bins=15,
            #bins=bins[i],
            #orientation='horizontal',
            range=spans[info],
        )
        axes[i].text(0.8, 0.9, "True age = {} Myr".format(ages[i]),
                     fontsize=12,
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=axes[i].transAxes
        )
        # axes[i].set_title(ages[i])
        # axes[i].legend(loc='best')
    f.set_tight_layout(tight=True)
    f.savefig('temp_plots/multi_{}_hist.pdf'.format(info))
#
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
