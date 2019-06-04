from __future__ import print_function, division

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, '..')
import chronostar.fitplotter as fp
import chronostar.tabletool as tt
from chronostar import component

def plotEveryIter(rdir, star_pars, bg_hists=None, true_memb=None):
    try:
        dims_set = ('xy', 'uv', 'xu', 'yv', 'zw', 'xw')
        print("Attempting init")
        if os.path.isfile(rdir + 'init_{}.pdf'.format(dims_set[-1])):
            print("  init already plotted...")
        else:
            for dim1, dim2 in dims_set:
                plt.clf()
                fp.plotPane(dim1, dim2, star_pars=star_pars,
                            comps=rdir + 'init_comps.npy',
                            comp_now=True,
                            membership=true_memb)
                plt.savefig(rdir + 'init_{}{}.pdf'.format(dim1, dim2))
    except:
        print("init lacking files")

    iter_count = 0
    while True:
        try:
            print("Attempting iter {}".format(iter_count))
            # idir = rdir + 'iter{}/'.format(iter_count)
            idir = rdir + 'iter{:02}/'.format(iter_count)
            if os.path.isfile(idir + 'iter_{:02}_xw.pdf'.format(iter_count)):
                print('    iter_{:02} already plotted'.format(iter_count))
            else:
                z = np.load(idir + 'membership.npy')
                weights = z.sum(axis=0)
                for dim1, dim2 in ('xy', 'uv', 'xu', 'yv', 'zw', 'xw'):
                    plt.clf()
                    fp.plotPane(dim1, dim2, star_pars=star_pars,
                                comps=idir + 'best_comps.npy',
                                comp_now=True,
                                membership=z,
                                )
                    plt.savefig(idir + 'iter_{:02}_{}{}.pdf'.format(
                        iter_count, dim1, dim2))
            iter_count += 1
        except IOError:
            print("Iter {} is lacking files".format(iter_count))
            break
    try:
        print("Attempting final")
        idir = rdir + 'final/'
        if os.path.isfile(idir + 'final_xw.pdf'):
            print("    final already plotted")
        else:
            z = np.load(idir + 'final_membership.npy')
            weights = z.sum(axis=0)
            for dim1, dim2 in ('xy', 'uv', 'xu', 'yv', 'zw', 'xw'):
                plt.clf()
                fp.plotPaneWithHists(dim1, dim2, star_pars=star_pars,
                                     comps=idir + 'final_groups.npy',
                                     comp_now=True,
                                     membership=z,
                                     true_memb=true_memb,
                                     )
                plt.savefig(idir + 'final_{}{}.pdf'.format(
                    dim1, dim2))
    except IOError:
        print("final is lacking files")
    return

def getZfromOrigins(origins, star_pars):
    if type(origins) is str:
        origins = SphereComponent.load_components(origins)
    if type(star_pars) is str:
        star_pars = tt.build_data_dict_from_table(star_pars)
    nstars = star_pars['means'].shape[0]
    ngroups = len(origins)
    nassoc_stars = np.sum([o.nstars for o in origins])
    using_bg = nstars != nassoc_stars
    z = np.zeros((nstars, ngroups + using_bg))
    stars_so_far = 0
    # set associaiton members memberships to 1
    for i, o in enumerate(origins):
        z[stars_so_far:stars_so_far+o.nstars, i] = 1.
        stars_so_far += o.nstars
    # set remaining stars as members of background
    if using_bg:
        z[stars_so_far:,-1] = 1.
    return z

rdir = sys.argv[1]
rdir = rdir.rstrip('/') + '/'
data_file = sys.argv[2]
if len(sys.argv) > 3:
    comp_choice = sys.argv[3]
    Component = component.IMPLEMNTATION.get(comp_choice,
                                            component.SphereComponent)
    print('Using {} implementation'.format(Component))

# try:
#     label = sys.argv[2]
# except IndexError:
#     label = None
# star_pars_file = '../data/{}_xyzuvw.fits'.format(assoc_name)

if os.path.isfile(rdir + 'bg_hists.npy'):
    bg_hists = np.load(rdir + 'bg_hists.npy')
else:
    bg_hists = None

# First, if stars are synthetic, plot true groups
# true_memb = None
# if is_synth_fit:
#     origins = np.load(rdir + 'synth_data/origins.npy')
#     true_memb = getZfromOrigins(origins, star_pars_file)
#     with_bg = len(origins) < true_memb.shape[1]
#     assert true_memb.shape[0] == dt.loadXYZUVW(star_pars_file)['xyzuvw'].shape[0]
#     if len(origins.shape) == 0:
#         origins = np.array(origins.item())
#     weights = np.array([origin.nstars for origin in origins])
#     for dim1, dim2 in ('xy', 'uv', 'xu', 'yv', 'zw', 'xw'):
#         plt.clf()
#         fp.plotPaneWithHists(dim1, dim2, star_pars=star_pars_file,
#                              groups=origins, weights=weights,
#                              group_now=True, with_bg=with_bg,
#                              no_bg_covs=with_bg,
#                              )
#         plt.savefig(rdir + 'pre_plot_{}{}.pdf'.format(dim1,dim2))

# Now choose if handling incremental fit or plain fit
true_memb = None
ncomps = 1
if type(data_file) is str:
    star_pars = tt.build_data_dict_from_table(data_file)
print("nstars: {}".format(star_pars['means'].shape[0]))
while os.path.isdir(rdir + '{}/'.format(ncomps)):
    print("ncomps: {}".format(ncomps))
    if ncomps == 1:
        plotEveryIter(rdir + '{}/'.format(ncomps), star_pars, bg_hists,
                      true_memb=true_memb)
    else:
        for i in range(ncomps-1):
            print("sub directory {}".format(chr(ord('A') + i)))
            subrdir = rdir + '{}/{}/'.format(ncomps, chr(ord('A') + i))
            if os.path.isdir(subrdir):
                plotEveryIter(subrdir, star_pars, bg_hists,
                              true_memb=true_memb)
    ncomps += 1





