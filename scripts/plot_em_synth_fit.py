from __future__ import print_function, division

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, '..')
import chronostar.fitplotter as fp


def plotEveryIter(rdir, star_pars):
    iter_count = 0
    while True:
        try:
            print("Attempting iter {}".format(iter_count))
            idir = rdir + 'iter{}/'.format(iter_count)
            z = np.load(idir + 'membership.npy')
            weights = z.sum(axis=0)
            for dim1, dim2 in ('xy', 'uv', 'xu', 'yv', 'zw', 'xw'):
                plt.clf()
                fp.plotPaneWithHists(dim1, dim2, star_pars=star_pars,
                                     groups=idir + 'best_groups.npy',
                                     weights=weights, group_now=True)
                plt.savefig(idir + 'iter_{}_{}{}.pdf'.format(
                    iter_count, dim1, dim2))
            iter_count += 1
        except IOError:
            print("Iter {} is lacking files".format(iter_count))
            break
    try:
        print("Attempting final")
        idir = rdir + 'final/'
        z = np.load(idir + 'final_membership.npy')
        weights = z.sum(axis=0)
        for dim1, dim2 in ('xy', 'uv', 'xu', 'yv', 'zw', 'xw'):
            plt.clf()
            fp.plotPaneWithHists(dim1, dim2, star_pars=star_pars,
                                 groups=idir + 'final_groups.npy',
                                 weights=weights, group_now=True)
            plt.savefig(idir + 'iter_{}_{}{}.pdf'.format(
                iter_count, dim1, dim2))
    except IOError:
        print("final is lacking files")
    return

assoc_name = sys.argv[1]
star_pars_file = '../data/{}_xyzuvw.fits'.format(assoc_name)
rdir = '/data/mash/tcrun/em_fit/{}/'.format(assoc_name)
if not os.path.isdir(rdir):
    rdir = '../results/em_fit/{}/'.format(assoc_name)

is_inc_fit = os.path.isdir(rdir + '1/')
is_synth_fit = os.path.isdir(rdir + 'synth_data/')

# if stars are synthetic, plot true groups
if is_synth_fit:
    origins = np.load(rdir + 'synth_data/origins.npy')
    if len(origins.shape) == 0:
        origins = np.array(origins.item())
    weights = np.array([origin.nstars for origin in origins])
    for dim1, dim2 in ('xy', 'uv', 'xu', 'yv', 'zw', 'xw'):
        plt.clf()
        fp.plotPaneWithHists(dim1, dim2, star_pars=star_pars_file,
                             groups=origins,
                             weights=weights, group_now=True)
        plt.savefig(rdir + 'pre_plot_{}{}.pdf'.format(dim1,dim2))

if not is_inc_fit:
    plotEveryIter(rdir, star_pars_file)

else: # incremental fit
    ncomps = 1
    while os.path.isdir(rdir + '{}/'.format(ncomps)):
        print("ncomps: {}".format(ncomps))
        if ncomps == 1:
            plotEveryIter(rdir + '{}/'.format(ncomps), star_pars_file)
        else:
            for i in range(ncomps-1):
                print("sub directory {}".format(chr(ord('A') + i)))
                subrdir = rdir + '{}/{}/'.format(ncomps, chr(ord('A') + i))
                if os.path.isdir(subrdir):
                    plotEveryIter(subrdir, star_pars_file)
        ncomps += 1





