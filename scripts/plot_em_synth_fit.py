from __future__ import print_function, division

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, '..')
import chronostar.fitplotter as fp

assoc_name = sys.argv[1]
star_pars_file = '../data/{}_xyzuvw.fits'.format(assoc_name)
rdir = '/data/mash/tcrun/em_fit/{}/'.format(assoc_name)
if not os.path.isdir(rdir):
    rdir = '../results/em_fit/{}/'.format(assoc_name)

is_inc_fit = os.path.isdir(rdir + '1/')

if not is_inc_fit:
    iter_count = 0
    while True:
        try:
            print("Attempting iter {}".format(iter_count))
            idir = rdir + 'iter{}/'.format(iter_count)
            z = np.load(rdir + 'membership.npy')
            weights = z.sum(axis=0)
            for dim1, dim2 in ('xy', 'uv', 'xu', 'yv', 'zw', 'xw'):
                plt.clf()
                fp.plotPaneWithHists(dim1, dim2, star_pars=star_pars_file,
                                     groups=idir+'best_groups.npy',
                                     weights=weights, group_now=True)
                plt.savefig(idir + 'iter_{}_{}{}.pdf'.format(
                    iter_count,dim1,dim2))
            iter_count += 1
        except IOError:
            print("Iter {} is lacking files".format(iter_count))
            break




