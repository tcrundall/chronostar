from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '..')
import chronostar.groupfitter as gf
import chronostar.synthesiser as syn
import chronostar.datatool as dt


rdir = 'example_comps/'
init_groups = dt.loadGroups(rdir+'iter00/best_groups.npy')
init_z = np.load(rdir+'iter00/membership.npy')
later_groups = dt.loadGroups(rdir + 'iter15/best_groups.npy')
later_z = np.load(rdir+'iter15/membership.npy')

xs = np.linspace(2, 100,10000)
plt.clf()
for sig in [.5, .75, 1]:
    plt.plot(xs, gf.lnlognormal2(xs, sig=sig), label='{}'.format(sig))
plt.legend(loc='best')
plt.savefig("lnlognormal.pdf")

print("Init alphas")
for i, (igroup, lgroup) in enumerate(zip(init_groups, later_groups)):
    sig=1.
    print('Group {}'.format(i))
    print('Init alpha: {:.5} | Later alpha: {:.5}'.format(
        gf.calcAlpha(igroup.dx, igroup.dv, init_z[:,i].sum()),
        gf.calcAlpha(lgroup.dx, lgroup.dv, later_z[:,i].sum())
    ))
    print('New prior, with sig = {}'.format(sig))
    print('Init prior: {:.5} | Later prior: {:.5}'.format(
        gf.lnAlphaPrior2(igroup.getInternalSphericalPars(), init_z[:,i], sig=sig),
        gf.lnAlphaPrior2(lgroup.getInternalSphericalPars(), later_z[:,i], sig=sig),
    ))
    print('Retired prior:')
    print('Init prior: {:.5} | Later prior: {:.5}'.format(
        gf.lnAlphaPrior(igroup.getInternalSphericalPars(), None, init_z[:,i]),
        gf.lnAlphaPrior(lgroup.getInternalSphericalPars(), None, later_z[:,i]),
    ))

gaia_6d_hist_file = 'gaia_6d_hist.npy'
try:
    gaia_6d_hist = np.load(gaia_6d_hist_file)
    print("Loaded histogram with bins {}".format(gaia_6d_hist[0].shape[0]))
except IOError:
    bins = 20
    print("Generating histogram with bins {}".format(bins))
    gaia_xyzuvw = np.load('../data/gaia_dr2_mean_xyzuvw.npy')
    gaia_6d_hist = np.histogramdd(gaia_xyzuvw, bins=bins)
    np.save(gaia_6d_hist_file, gaia_6d_hist)

near_g6d_hist_file = 'near_g6d_hist.npy'
offset = np.array([ 200., 200., 100., 10., 10., 10.])
try:
    raise IOError
    near_g6d_hist = np.load(near_g6d_hist_file)
    print("Loaded near histogram with bins {}".format(near_g6d_hist[0].shape[0]))
except IOError:
    bins = 5
    print("Generating near histogram with bins {}".format(bins))
    gaia_xyzuvw = np.load('../data/gaia_dr2_mean_xyzuvw.npy')
    ref_mean = init_groups[0].mean
    near_mask = np.where(
        np.all( (gaia_xyzuvw < ref_mean+offset)
                & (gaia_xyzuvw > ref_mean-offset),
                axis=1)
        )
    near_gaia = gaia_xyzuvw[near_mask]
    near_g6d_hist = np.histogramdd(near_gaia, bins=bins)
    np.save(near_g6d_hist_file, near_g6d_hist)



ref_mean = init_groups[0].mean
print("Typical density {}".format(dt.calcHistogramDensity(
    ref_mean, gaia_6d_hist[0], gaia_6d_hist[1]
)))
print("Typical sensitive density {}".format(dt.calcHistogramDensity(
    ref_mean, near_g6d_hist[0], near_g6d_hist[1]
)))

print("Peak densities of components")
for i, (igroup, lgroup) in enumerate(zip(init_groups, later_groups)):
    print("Group ", i)
    print("Init peak: {} | Later peak: {}".format(
    dt.mvGauss(init_groups[i].mean, init_groups[i].mean,
               init_groups[i].generateSphericalCovMatrix(),
               init_z[:,i].sum()),
    dt.mvGauss(later_groups[i].mean, later_groups[i].mean,
               later_groups[i].generateSphericalCovMatrix(),
               later_z[:,i].sum())
    ))

