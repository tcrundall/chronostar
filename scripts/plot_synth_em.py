from __future__ import print_function, division
'''Generate a diagram detailing the result of an EM fit to data'''

from distutils.dir_util import mkpath
import itertools
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
import chronostar.synthesiser as syn
import chronostar.traceorbit as torb
import chronostar.measurer as ms
import chronostar.converter as cv
import chronostar.datatool as dt
import chronostar.fitplotter as fp


assoc_name = sys.argv[1]


pdir = '../figures/{}/'.format(assoc_name)
mkpath(pdir)
rdir = '../results/em_fit/{}/'.format(assoc_name)

try:
    assoc_name_split = assoc_name.split('_')
    int(assoc_name_split[-1])
    xyzuvw_file = '../data/{}_xyzuvw.fits'.format('_'.join(assoc_name_split[:-1]))
except:
    xyzuvw_file = '../data/{}_xyzuvw.fits'.format(assoc_name)

ERROR = 1.0

star_pars = dt.loadXYZUVW(xyzuvw_file)
final_groups = np.load(rdir + 'final/final_groups.npy')
final_z = np.load(rdir + 'final/final_membership.npy')

final_weights = final_z.sum(axis=0)

dims = 'XYZUVW'

# For each combination, plot an annotated schematic of a single component
for i, dim1 in enumerate(dims):
    for dim2 in dims[i+1:]:
        plt.clf()
        ax = plt.subplot()
        print(dim1, dim2)
        ax.set_title('{} and {}'.format(dim1, dim2))
        fp.plotPaneWithHists(dim1, dim2, groups=final_groups, weights=final_weights,
                             star_pars=star_pars, group_then=True, group_now=True,
                             group_orbit=True)
        # fp.plotPane(dim1, dim2, ax=ax, groups=final_groups, star_pars=star_pars,
        #             group_then=True, group_now=True, group_orbit=True,
        #             annotate=True)
        plt.savefig(pdir + 'schematic-{}{}.pdf'.format(dim1, dim2),
                    bbox_inches='tight')

# Just some fun... every possible pair on the one plot
dim_pairs = list(itertools.combinations(dims, 2))
fp.plotMultiPane(dim_pairs, star_pars, final_groups, save_file=pdir + 'all.pdf')


