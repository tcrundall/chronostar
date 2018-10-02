"""
Provided some set of stars (typically an association) snip all gaia
stars nearby (pos and vel)
"""
from __future__ import print_function, division

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '..')
import chronostar.datatool as dt

MARGIN = 2.0

assoc_name = "bpmg_cand_w_gaia_dr2_astrometry_comb_binars"
xyzuvw_file = "../data/{}_xyzuvw.fits".format(assoc_name)
xyzuvw_dict = dt.loadXYZUVW(xyzuvw_file)

# construct box
kin_max = np.max(xyzuvw_dict['xyzuvw'], axis=0)
kin_min = np.min(xyzuvw_dict['xyzuvw'], axis=0)
span = kin_max - kin_min
upper_boundary = kin_max + MARGIN*span
lower_boundary = kin_min - MARGIN*span


# get gaia stars within box
gaia_xyzuvw_file = "../data/gaia_dr2_mean_xyzuvw.npy"
gaia_xyzuvw = np.load(gaia_xyzuvw_file)
mask = np.where(
    np.all(
        (gaia_xyzuvw < upper_boundary) & (gaia_xyzuvw > lower_boundary),
        axis=1)
)
print(mask[0].shape)
nearby_gaia = gaia_xyzuvw[mask]


# plot histograms
labels = 'XYZUVW'
for i, label in enumerate(labels):
    plt.clf()
    plt.hist(nearby_gaia[:,i], bins=int(MARGIN**1.5*25))
    plt.xlabel(label)
    plt.savefig("temp_plots/nearby_gaia_{}.pdf".format(label))

hists = []
bins = int(MARGIN**1.5 * 25)
n_nearby = nearby_gaia.shape[0]
norm = n_nearby ** (5./6)
for col in nearby_gaia.T:
    hists.append(np.histogram(col, bins))
# fit gaus and flat line to distros

mystar = nearby_gaia[0]
lnlike_star = 0
for i in range(6):
    # DIGITIZE SEEMS TO RETURN DESIRED INDEX + 1...?
    lnlike_star += np.log(hists[i][0][np.digitize(mystar[i], hists[i][1])-1])
lnlike_star -= n_nearby**5


manual_ub = np.array([ 100., 50., 40., 8., 0., 2.])
manual_lb = np.array([-100.,-70.,-30.,-6.,-7.,-7.])

man_mask = np.where(
    np.all(
        (gaia_xyzuvw < manual_ub) & (gaia_xyzuvw > manual_lb),
        axis=1)
)
print(man_mask[0].shape)
man_nearby_gaia = gaia_xyzuvw[man_mask]


# plot histograms
labels = 'XYZUVW'
for i, label in enumerate(labels):
    plt.clf()
    plt.hist(man_nearby_gaia[:,i], bins=20)
    plt.xlabel(label)
    plt.savefig("temp_plots/man_nearby_gaia_{}.pdf".format(label))

man_hists = []
bins = 20
man_n_nearby = man_nearby_gaia.shape[0]
# norm = n_nearby ** (5./6)
for col in nearby_gaia.T:
    man_hists.append(np.histogram(col, bins))
#
hist6d = np.histogramdd(man_nearby_gaia, bins=3)

# for nbins in range(10,40):
#     all_hist6d = np.histogramdd(gaia_xyzuvw, bins=nbins)
#     bin_widths = [bins[1] - bins[0] for bins in all_hist6d[1]]
#     bin_area = np.prod(bin_widths)
#
#     bpmean = np.array([0,0,0,0,-4,-2])
#     bp_ix = tuple([np.digitize(bpmean[dim],all_hist6d[1][dim]) - 1
#                    for dim in range(6)])
#     density_near_bp = all_hist6d[0][bp_ix] / bin_area
#     print("{:02}: {:.5} | {}".format(nbins,density_near_bp), bin_widths)

for nbins in range(2,10):
    near_hist6d = np.histogramdd(nearby_gaia, bins=nbins)
    bin_widths = [bins[1] - bins[0] for bins in near_hist6d[1]]
    bin_area = np.prod(bin_widths)

    bpmean = np.array([0,0,0,0,-4,-2])
    bp_ix = tuple([np.digitize(bpmean[dim],near_hist6d[1][dim]) - 1
                   for dim in range(6)])
    density_near_bp = near_hist6d[0][bp_ix] / bin_area
    print("{:02}: {:.5} | {}".format(nbins,density_near_bp, bin_widths))

# for dim, label in enumerate(labels):
#     plt.clf()
#     plt.bar(hist6d[1][dim],np.sum)
#     plt.hist(man_nearby_gaia[:,dim], bins=20)
#     plt.xlabel(label)
#     plt.savefig("temp_plots/man_nearby_gaia_{}.pdf".format(label))
