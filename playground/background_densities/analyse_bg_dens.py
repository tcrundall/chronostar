from __future__ import division, print_function

"""
Playing around with different approaches for finding typical density near
BPMG
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys
sys.path.insert(0, '..')
import chronostar.datatool as dt
import chronostar.synthesiser as syn
import chronostar.expectmax as em
import chronostar.groupfitter as gf
from chronostar._overlap import get_lnoverlaps

bpmg_group_file = 'final_groups.npy'
gaia_sep_hist_file = 'bg_hists.npy'
gaia_6d_hist_file = 'gaia_6d_hist.npy'
gaia_xyzuvw = np.load('../data/gaia_dr2_mean_xyzuvw.npy')

bpmg_group = dt.loadGroups(bpmg_group_file)[0]
ref_mean = bpmg_group.mean[:]
# ref_mean[-1] += 4. #SANITY CHECK, perform same analysis for W offset by 4
print("1 comp. fit to BPMG has mean {}".format(ref_mean))
print("We will calculate all densities at this point...")
gaia_sep_hist = np.load(gaia_sep_hist_file)
try:
    gaia_6d_hist = np.load(gaia_6d_hist_file)
    print("Loaded histogram with bins {}".format(gaia_6d_hist[0].shape[0]))
except IOError:
    bins = 20
    print("Generating histogram with bins {}".format(bins))
    gaia_xyzuvw = np.load('../data/gaia_dr2_mean_xyzuvw.npy')
    gaia_6d_hist = np.histogramdd(gaia_xyzuvw, bins=bins)
    np.save(gaia_6d_hist_file, gaia_6d_hist)

# near_g6d_hist_file = 'near_g6d_hist.npy'
# offset = np.array([ 500., 500., 200., 30., 30., 20.])
# near_mask = np.where(
#     np.all( (gaia_xyzuvw < ref_mean+offset)
#             & (gaia_xyzuvw > ref_mean-offset),
#             axis=1)
# )
# near_gaia = gaia_xyzuvw[near_mask]
#
# # density as calculated by 6 separate 1D histograms (Mike's initial suggestion)
# typ_sep_bpmg_dens = np.exp(em.backgroundLogOverlap(ref_mean, gaia_sep_hist))
# print("Density from 6 separate 1D hists: {}".format(typ_sep_bpmg_dens))
#
# # densities as calculated by 1 6D histogram across vicinity, (varying bin size)
# all_bpmg_dens = []
# bins = np.arange(1,30)
# print("Generating hist of nearby stars with bins")
# for bin in bins:
#     print(".. {}".format(bin))
#     near_g6d_hist = np.histogramdd(near_gaia, bins=bin)
#     # np.save(near_g6d_hist_file, near_g6d_hist)
#     all_bpmg_dens.append(dt.calcHistogramDensity(ref_mean, near_g6d_hist[0],
#                                              near_g6d_hist[1]))
#
# all_bpmg_dens = np.array(all_bpmg_dens)
# typ_bpmg_dens = np.mean(all_bpmg_dens[np.where(all_bpmg_dens != 0.0)][-6:])
# typ_bpmg_dens = np.mean(all_bpmg_dens[-10:])
# plt.clf()
# plt.plot(bins, all_bpmg_dens, label='1 6D histogram')
# plt.plot(bins, len(bins)*[typ_bpmg_dens], label='Typical 6D histograms')
# plt.plot(bins, len(bins)*[typ_sep_bpmg_dens], label='6 1D histograms')
# plt.xlabel('bin count')
# plt.ylabel('density at bpmg mean')
# plt.legend(loc='best')
# plt.savefig("dens-vs-bincount.pdf")
# plt.yscale('log')
# plt.savefig("log-dens-vs-bincount.pdf")
# print("Typical density of 6D histogram: {}".format(typ_bpmg_dens))

gaia_std = np.std(gaia_xyzuvw, axis=0)

bpmg_mean = np.array(bpmg_group.mean)
twin_mean = bpmg_mean.copy()
twin_mean[2] *= -1
twin_mean[5] *= -1


if False:
    w_step = []
    x_step = []
    bins_per_std = []
    bpmg_gen_dens = []
    twin_gen_dens = []

    for i in range(3,20):
        bins_per_std.append(i)
        w_step.append(gaia_std[-1]/float(i))
        x_step.append(gaia_std[0]/i)
        bpmg_gen_dens.append(dt.getDensity(bpmg_mean, gaia_xyzuvw, i))
        twin_gen_dens.append(dt.getDensity(twin_mean, gaia_xyzuvw, i))

    plt.clf()
    plt.plot(x_step, bpmg_gen_dens, label='bpmg mean')
    plt.plot(x_step, twin_gen_dens, label='twin mean')
    plt.xlabel('x step [pc]')
    plt.ylabel(r'density [pc km/s]$^{-3}$')
    plt.yscale('log')
    plt.title('Density at point given different box size')
    plt.legend(loc='best')
    plt.xlim(plt.xlim()[::-1])
    plt.savefig('dens-vs-x-step.pdf')

    plt.clf()
    plt.plot(w_step, bpmg_gen_dens, label='bpmg mean')
    plt.plot(w_step, twin_gen_dens, label='twin mean')
    plt.xlabel('w step [km/s]')
    plt.ylabel(r'density [pc km/s]$^{-3}$')
    plt.yscale('log')
    plt.title('Density at point given different box size')
    plt.legend(loc='best')
    plt.xlim(plt.xlim()[::-1])
    plt.savefig('dens-vs-w-step.pdf')

    plt.clf()
    plt.plot(bins_per_std, bpmg_gen_dens, label='bpmg mean')
    plt.plot(bins_per_std, twin_gen_dens, label='twin mean')
    plt.xlabel('steps per gaia standard deviation')
    plt.ylabel(r'density [pc km/s]$^{-3}$')
    plt.yscale('log')
    plt.title('Density at point given different box size')
    plt.legend(loc='best')
    plt.savefig('dens-vs-bins-per-std.pdf')



# Investigate density of components for reference
rdir = 'example_comps/'
init_groups = dt.loadGroups(rdir+'iter00/best_groups.npy')
init_z = np.load(rdir+'iter00/membership.npy')
later_groups = dt.loadGroups(rdir + 'iter15/best_groups.npy')
later_z = np.load(rdir+'iter15/membership.npy')

hists_from_run = np.load('bg_hists.npy')

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



print("bg dens during run for reference:")
print(np.exp(em.backgroundLogOverlap(init_groups[0].mean, hists_from_run)))

# --------------------------------------------------
# ----- an aside, does a delta function at ---------
# ----- group mean return same as peak val ---------
# --------------------------------------------------
# spoiler... yes, yes it does
lg0 = later_groups[0]
delta_ol_at_mean =\
    np.exp(get_lnoverlaps(lg0.generateSphericalCovMatrix(), lg0.mean,
                          np.array([1e-10*np.eye(6)]), np.array([lg0.mean]),
                          1))[0]
peak_dens = dt.mvGauss(lg0.mean, lg0.mean, lg0.generateSphericalCovMatrix())
assert np.isclose(delta_ol_at_mean, peak_dens)

# demo synthesising some bg stars
my_bg_dens = 1e-7

spread = np.std(gaia_xyzuvw, axis=0) / 8.
ubound = bpmg_mean + 0.5*spread
lbound = bpmg_mean - 0.5*spread
nbg_stars = int(my_bg_dens * np.prod(spread))
print("There are {} stars in bg".format(nbg_stars))
my_bg_stars = np.random.uniform(-1,1,size=(nbg_stars, 6)) * spread + bpmg_mean

plt.clf()
plt.plot(my_bg_stars[:,0], my_bg_stars[:,5], '.')
plt.plot(bpmg_mean[0], bpmg_mean[5], 'x')
plt.xlabel("X [pc]")
plt.ylabel("W [km/s]")
plt.savefig('demo_synth_bg_stars.pdf')

# Using scipy kernel density estimator
X, Y, Z, U, V, W = np.mgrid[lbound[0]:lbound[0]:10j, lbound[1]:ubound[1]:10j, lbound[2]:ubound[2]:10j, lbound[3]:ubound[3]:10j, lbound[4]:ubound[4]:10j, lbound[5]:ubound[5]:10j]
positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel(), U.ravel(), V.ravel(), W.ravel()])
kernel = stats.gaussian_kde(gaia_xyzuvw.T)
print("kernel at BPMG mean: {}".format(kernel(bpmg_mean)[0]*gaia_xyzuvw.shape[0]))
print("kernel at TWIN mean: {}".format(kernel(twin_mean)[0]*gaia_xyzuvw.shape[0]))

farubound = bpmg_mean + 2.*spread
farlbound = bpmg_mean - 2.*spread
nearish_stars = gaia_xyzuvw[np.where(np.all((gaia_xyzuvw > farlbound)
                                            & (gaia_xyzuvw < farubound),
                                            axis=1))]

nearby_stars = gaia_xyzuvw[np.where(np.all((gaia_xyzuvw > lbound)
                                           & (gaia_xyzuvw < ubound),
                                           axis=1))]
nearish_kernel = stats.gaussian_kde(nearish_stars.T)
print("nearish kernel at BPMG mean: {}".format(nearish_kernel(bpmg_mean)[0]*nearish_stars.shape[0]))
near_kernel = stats.gaussian_kde(nearby_stars.T)
print("near kernel at BPMG mean: {}".format(near_kernel(bpmg_mean)[0]*nearby_stars.shape[0]))

near_kernel_silv = stats.gaussian_kde(nearby_stars.T, bw_method='silverman')
print("near kernel silv at BPMG mean: {}".format(near_kernel_silv(bpmg_mean)[0]*nearby_stars.shape[0]))

def scotts_rule(shape):
    npoints = shape[0]
    ndim = shape[1]
    return npoints**(-1./(ndim+4))


def get_density_from_kernel(stars, span_per_std, centre, test_point=None,
                            print_nstars=False):
    if test_point is None:
        test_point = centre
    gaia_std = np.std(stars, axis=0)
    spread = gaia_std / float(span_per_std)
    ubound = centre + 0.5*spread
    lbound = centre - 0.5*spread

    subset = stars[
        np.where(np.all((stars > lbound) & (stars < ubound), axis=1))
    ]
    nstars = subset.shape[0]
    kernel = stats.gaussian_kde(subset.T)
    if print_nstars:
        print("-- nstars: {}".format(nstars))

    return kernel.evaluate(test_point)[0] * nstars

if True:
    w_step = []
    x_step = []
    spread_per_std = []
    # bins_per_std = []
    bpmg_gen_dens = []
    twin_gen_dens = []
    twin_from_bpmg_dens = []

    gaia_std = np.std(gaia_xyzuvw, axis=0)
    print("Gaia std: {}".format(gaia_std))
    print("Steps per std\tx_step\tw_step\tnstars\tnormed dens\tscaled dens\t"
          "scaled dens 2")
    for i in [0.1,0.25,0.5,0.75, 1, 3, 5, 7, 9, 11]:
        spread_per_std.append(i)
        spread = gaia_std / float(i)
        ubound = bpmg_mean + 0.5*spread
        lbound = bpmg_mean - 0.5*spread
        subset = gaia_xyzuvw[np.where(
            np.all((gaia_xyzuvw > lbound) & (gaia_xyzuvw < ubound), axis=1)
        )]
        nstars = subset.shape[0]
        x_step.append(spread[0])
        w_step.append(spread[-1])
        # w_step.append(gaia_std[-1]/float(i))
        # x_step.append(gaia_std[0]/i)
        bpmg_kernel = stats.gaussian_kde(subset.T)
        norm_bpmg_dens = bpmg_kernel.evaluate(bpmg_mean)[0]
        bpmg_dens = norm_bpmg_dens * nstars
        bpmg_dens_2 = get_density_from_kernel(gaia_xyzuvw, i,
                                              bpmg_mean)
        print("{}\t\t{:.5}\t{:.5}\t{}\t{:.4}\t{:.4}\t{:.4}".format(
            i, x_step[-1], w_step[-1], nstars, norm_bpmg_dens, bpmg_dens,
            bpmg_dens_2
        ))
        # bpmg_gen_dens.append(bpmg_kernel.evaluate(bpmg_mean)[0] * nstars)


        bpmg_gen_dens.append(get_density_from_kernel(gaia_xyzuvw,
                                                     i, bpmg_mean))
        print("-- Twin kernel ", end='')
        twin_gen_dens.append(get_density_from_kernel(gaia_xyzuvw,
                                                     i, twin_mean,
                                                     print_nstars=True))
        twin_from_bpmg_dens.append(get_density_from_kernel(
            gaia_xyzuvw, i, bpmg_mean, twin_mean
        ))

    plt.clf()
    plt.plot(x_step, bpmg_gen_dens, label='bpmg mean')
    plt.plot(x_step, twin_gen_dens, label='twin mean')
    plt.plot(x_step, twin_from_bpmg_dens, label='twin mean bpmg')
    plt.xlabel('x step [pc]')
    plt.ylabel(r'density [pc km/s]$^{-3}$')
    plt.yscale('log')
    plt.title('Density at point given different box size')
    plt.legend(loc='best')
    plt.xlim(plt.xlim()[::-1])
    plt.savefig('dens-vs-x-step.pdf')

    plt.clf()
    plt.plot(w_step, bpmg_gen_dens, label='bpmg mean')
    plt.plot(w_step, twin_gen_dens, label='twin mean')
    plt.plot(w_step, twin_from_bpmg_dens, label='twin mean bpmg')
    plt.xlabel('w step [km/s]')
    plt.ylabel(r'density [pc km/s]$^{-3}$')
    plt.yscale('log')
    plt.title('Density at point given different box size')
    plt.legend(loc='best')
    plt.xlim(plt.xlim()[::-1])
    plt.savefig('dens-vs-w-step.pdf')

    plt.clf()
    plt.plot(spread_per_std, bpmg_gen_dens, label='bpmg mean')
    plt.plot(spread_per_std, twin_gen_dens, label='twin mean')
    plt.plot(spread_per_std, twin_from_bpmg_dens, label='twin mean bpmg')
    plt.xlabel('steps per gaia standard deviation')
    plt.ylabel(r'density [pc km/s]$^{-3}$')
    plt.yscale('log')
    plt.title('Density at point given different box size')
    plt.legend(loc='best')
    plt.savefig('dens-vs-bins-per-std.pdf')
