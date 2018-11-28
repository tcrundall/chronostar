from __future__ import print_function, division

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, '..')
import chronostar.fitplotter as fp
import chronostar.datatool as dt


def plotEveryIter(rdir, star_pars, bg_hists=None, true_memb=None):
    try:
        print("Attempting init")
        if os.path.isfile(rdir + 'init_xw.pdf'):
            print("  init already plotted...")
        else:
            for dim1, dim2 in ('xy', 'uv', 'xu', 'yv', 'zw', 'xw'):
                plt.clf()
                fp.plotPaneWithHists(dim1, dim2, star_pars=star_pars,
                                     groups=rdir + 'init_groups.npy',
                                     weights=None, group_now=True,
                                     bg_hists=bg_hists, membership=true_memb)
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
                    fp.plotPaneWithHists(dim1, dim2, star_pars=star_pars,
                                         groups=idir + 'best_groups.npy',
                                         weights=weights, group_now=True,
                                         bg_hists=bg_hists,
                                         membership=z,
                                         true_memb=true_memb
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
                                     groups=idir + 'final_groups.npy',
                                     weights=weights, group_now=True,
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
        origins = dt.loadGroups(origins)
    if type(star_pars) is str:
        star_pars = dt.loadXYZUVW(star_pars)
    nstars = star_pars['xyzuvw'].shape[0]
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


assoc_name = sys.argv[1]
try:
    label = sys.argv[2]
except IndexError:
    label = None
star_pars_file = '../data/{}_xyzuvw.fits'.format(assoc_name)
if not os.path.isfile(star_pars_file):
    banyan_data = '../data/gagne_bonafide_full_kinematics_with_lit' \
                  '_and_best_radial_velocity_comb_binars.fits'
    star_pars_file = dt.loadDictFromTable(banyan_data, assoc_name)
    print("Loaded association {} from banyan data".format(assoc_name))
    print("with total of {} stars".format(len(star_pars_file['xyzuvw'])))
rdir_suffix = 'em_fit/{}'.format(assoc_name.replace(' ','_'))
if label:
    rdir_suffix += '_{}/'.format(label)
else:
    rdir_suffix += '/'
rdir = '/data/mash/tcrun/' + rdir_suffix
if not os.path.isdir(rdir):
    rdir = '../results/' + rdir_suffix
print("Operating in directory: {}".format(rdir))

if os.path.isfile(rdir + 'bg_hists.npy'):
    bg_hists = np.load(rdir + 'bg_hists.npy')
else:
    bg_hists = None


is_inc_fit = os.path.isdir(rdir + '1/')
is_synth_fit = os.path.isdir(rdir + 'synth_data/')

# First, if stars are synthetic, plot true groups
true_memb = None
if is_synth_fit:
    origins = np.load(rdir + 'synth_data/origins.npy')
    true_memb = getZfromOrigins(origins, star_pars_file)
    assert true_memb.shape[0] == dt.loadXYZUVW(star_pars_file)['xyzuvw'].shape[0]
    if len(origins.shape) == 0:
        origins = np.array(origins.item())
    weights = np.array([origin.nstars for origin in origins])
    for dim1, dim2 in ('xy', 'uv', 'xu', 'yv', 'zw', 'xw'):
        plt.clf()
        fp.plotPaneWithHists(dim1, dim2, star_pars=star_pars_file,
                             groups=origins,
                             weights=weights, group_now=True)
        plt.savefig(rdir + 'pre_plot_{}{}.pdf'.format(dim1,dim2))

# Now choose if handling incremental fit or plain fit
if not is_inc_fit:
    plotEveryIter(rdir, star_pars_file)
else: # incremental fit
    ncomps = 1
    if type(star_pars_file) is str:
        star_pars_file = dt.loadXYZUVW(star_pars_file)
    print("nstars: {}".format(star_pars_file['xyzuvw'].shape[0]))
    while os.path.isdir(rdir + '{}/'.format(ncomps)):
        print("ncomps: {}".format(ncomps))
        if ncomps == 1:
            plotEveryIter(rdir + '{}/'.format(ncomps), star_pars_file, bg_hists,
                          true_memb=true_memb)
        else:
            for i in range(ncomps-1):
                print("sub directory {}".format(chr(ord('A') + i)))
                subrdir = rdir + '{}/{}/'.format(ncomps, chr(ord('A') + i))
                if os.path.isdir(subrdir):
                    plotEveryIter(subrdir, star_pars_file, bg_hists,
                                  true_memb=true_memb)
        ncomps += 1





