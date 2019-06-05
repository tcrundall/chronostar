import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from chronostar.component import SphereComponent
from chronostar.traceorbit import trace_cartesian_orbit
from chronostar import tabletool
import pickle

import numpy as np

def update_1D_lims(current_lims, recent_lims):
    if current_lims is None:
        return recent_lims
    else:
        try:
            return [np.min((current_lims[0], recent_lims[0])),
                    np.max((current_lims[1], recent_lims[1]))]
        except:
            import pdb; pdb.set_trace()

data_dir = '../data/synth_data_for_plot/'

true_pars = np.array([ -250., 1200.,-37., 24., -5., 5., 5., 1., 100.])
true_comp = SphereComponent(pars=true_pars)

# Initialising stumps
burnin_chain_shape = (18,0,9)
burnin_lnprob_shape = (18,0)
burnin_chain = np.zeros(burnin_chain_shape)
burnin_lnprob = np.zeros(burnin_lnprob_shape)

# Iteratively load and stitch together burnins
i = 0
while True:
    try:
        chain_segment = np.load(data_dir + 'burnin_chain{:02}.npy'.format(i))
        lnprob_segment = np.load(data_dir + 'burnin_lnprob{:02}.npy'.format(i))
        burnin_chain = np.concatenate((burnin_chain, chain_segment), axis=1)
        burnin_lnprob = np.concatenate((burnin_lnprob, lnprob_segment), axis=1)
    except IOError:
        break
    i += 1

print(burnin_chain.shape)
print(burnin_lnprob.shape)

nsteps = burnin_chain.shape[1]

# For each step, find the best component, and then plot it

# for step_ix in range(burnin_chain.shape[1]):
# lims = 6 * [None]

stride = 20
nplots = int(nsteps/stride)
print('Construction {} plots in total'.format(nplots))

# Some constants
dims = [(0,1), (0,3), (1,4), (2,5)]
labels = 'XYZUVW'
units = 3*['pc'] + 3*['km/s']
base_figure_file = 'base_figure.pkl'
star_data_file = data_dir + 'synth_for_plot_data.fit'

star_data = tabletool.build_data_dict_from_table(star_data_file)

# Set up base subplots, plotting everything that is the same across iterative
# plots. We will then store this via Pickle to save time
base_fig, base_ax = plt.subplots(nrows=2, ncols=2)
base_fig.set_size_inches(8,8)
base_fig.set_tight_layout(True)

lims = 6*[None]
for ax, (dim1, dim2) in zip(base_ax.flatten(), dims):
    true_comp.plot(ax=ax, dim1=dim1, dim2=dim2, comp_now=False, comp_then=True,
                   comp_orbit=True, linestyle='-.', color='grey', alpha=0.15)
    for star_mn, star_cov in zip(star_data['means'], star_data['covs']):
        star_comp = SphereComponent(attributes={
            'mean':star_mn,
            'covmatrix':star_cov,
            'age':0.
        })
        star_comp.plot(dim1=dim1, dim2=dim2, ax=ax, color='blue', comp_now=False,
                       comp_then=True, alpha=1., marker='.')
        lims[dim1] = update_1D_lims(lims[dim1], ax.get_xlim())
        lims[dim2] = update_1D_lims(lims[dim2], ax.get_ylim())

with open(base_figure_file, 'w') as fp:
    pickle.dump((base_fig, base_ax), fp)


for plot_ix in range(nplots):
    print('plot {} of {}'.format(plot_ix, nplots))
    step_ix = stride*plot_ix
    best_walker_ix = np.argmax(burnin_lnprob[:,step_ix])
    best_comp = SphereComponent(
            emcee_pars=burnin_chain[best_walker_ix, step_ix]
    )

    with open(base_figure_file, 'r') as fp:
        fig, ax = pickle.load(fp)
    # fig.set_size_inches(10,10)
    # fig.set_tight_layout(True)

    for sub_ax, (dim1, dim2) in zip(ax.flatten(), dims):
        print('subplot: {} {}'.format(dim1, dim2))
        best_comp.plot(ax=sub_ax, dim1=dim1, dim2=dim2, comp_now=True, comp_then=True,
                       comp_orbit=True, alpha=0.3)

        sub_ax.set_xlabel('{} [{}]'.format(labels[dim1], units[dim1]))
        sub_ax.set_ylabel('{} [{}]'.format(labels[dim2], units[dim2]))
        sub_ax.set_xlim(lims[dim1])
        # print('x set fine: {}'.format(lims))
        sub_ax.set_ylim(lims[dim2])
        # print('y set fine {}'.format(lims))

    # Put in some annotation
    first_ax = ax[0,0]
    first_ax.text(0.8, 0.9, '{:>5.1f} Myr  '.format(best_comp.get_age()),
                  horizontalalignment='right',
                  transform=first_ax.transAxes)
    first_ax.text(0.8, 0.8, '{:>6} steps'.format(step_ix),
                  horizontalalignment='right',
                  transform=first_ax.transAxes)

    plt.savefig('../plots/explanation_movie/{:04}_explanation.png'.\
        format(
            plot_ix,
            # labels[dim1],
            # labels[dim2]
        ))