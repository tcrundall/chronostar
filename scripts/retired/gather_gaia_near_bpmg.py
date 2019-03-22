from __future__ import print_function, division

"""
Goes through the Gaia (cartesian) data set, identifies all stars
near BPMG, removes duplicates (especially binaries/ternaries absorbed
into centre of mass references), and packages up neatly for an EM run
"""

from astropy.table import Table
import numpy as np
import sys
sys.path.insert(0, '..')
import chronostar.retired2.datatool as dt

cart_gaia_table = '../data/gaia_cartesian_full_6d_table.fits'
gagne_table = '../data/gagne_bonafide_full_kinematics_with_lit_and_best_'\
              'radial_velocity_comb_binars.fits'

assoc_name = 'beta Pictoris'
bp_star_pars = dt.loadDictFromTable(gagne_table, assoc_name=assoc_name)

print("Reading in gaia table...")
gt = Table.read(cart_gaia_table)
print("Gaia table read in")

MARGIN = 0.5
xyzuvw_file = "../data/{}_with_gaia_small_xyzuvw.fits".format(
    assoc_name.replace(' ','_')
)

# construct box
kin_max = np.max(bp_star_pars['xyzuvw'], axis=0)
kin_min = np.min(bp_star_pars['xyzuvw'], axis=0)
span = kin_max - kin_min
upper_boundary = kin_max + MARGIN*span
lower_boundary = kin_min - MARGIN*span

assert False

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

nearby_ids = gt['source_id'][mask]

if False:
    bp_source_ids = bp_star_pars['table']['source_id'][
        np.where(bp_star_pars['table']['Moving group'] == 'beta Pictoris')
    ]
    bp_table = bp_star_pars['table'][
        np.where(bp_star_pars['table']['Moving group'] == 'beta Pictoris')
    ]
    print("Building masks")
    intersec_ids = [id for id in nearby_ids if str(id) in bp_source_ids]
    non_intersec_ids = [id for id in nearby_ids if str(id) not in bp_source_ids]

    intersec_mask = [id not in intersec_ids for id in gt['source_id'][mask]]

    # take all non_intersec_ids stars, build new table, appending these onto
    # "bonafide" bpmg stars
    nearby_gaia_table = gt[mask][intersec_mask]

    nearby_gaia_table['source_id'] = nearby_gaia_table['source_id'].astype(np.str)

    print("Joining table")
    joined_table = table.vstack((bp_table, nearby_gaia_table))
    print(len(joined_table))
    #Table.write(joined_table, '../data/bpmg_gaia_all_cart_joined_xyzuvw.fits')
    Table.write(joined_table, xyzuvw_file)
    print("Table written")


# Now do the same thing but for whole BANYAN membership list
banyan_membs_source_ids = bp_star_pars['table']['source_id']
print("Building masks")
intersec_ids = [id for id in nearby_ids if str(id) in banyan_membs_source_ids]
non_intersec_ids = [id for id in nearby_ids if
                    str(id) not in banyan_membs_source_ids]

intersec_mask = [id not in intersec_ids for id in gt['source_id'][mask]]

# take all non_intersec_ids stars, build new table, appending these onto
# "bonafide" bpmg stars
nearby_gaia_table = gt[mask][intersec_mask]

nearby_gaia_table['source_id'] = nearby_gaia_table['source_id'].astype(np.str)

print("Joining table")
joined_table = table.vstack((bp_star_pars['table'], nearby_gaia_table))
print(len(joined_table))
#Table.write(joined_table, '../data/bpmg_gaia_all_cart_joined_xyzuvw.fits')
Table.write(joined_table, 'banyan_with_gaia_near_bpmg_xyzuvw.fits')
print("Table written")

# # plot histograms
# labels = 'XYZUVW'
# for i, label in enumerate(labels):
#     plt.clf()
#     plt.hist(nearby_gaia[:,i], bins=int(MARGIN**1.5*25))
#     plt.xlabel(label)
#     plt.savefig("temp_plots/nearby_gaia_{}.pdf".format(label))
# 
# hists = []
# bins = int(MARGIN**1.5 * 25)
# n_nearby = nearby_gaia.shape[0]
# norm = n_nearby ** (5./6)
# for col in nearby_gaia.T:
#     hists.append(np.histogram(col, bins))
# # fit gaus and flat line to distros
# 
# mystar = nearby_gaia[0]
# lnlike_star = 0
# for i in range(6):
#     # DIGITIZE SEEMS TO RETURN DESIRED INDEX + 1...?
#     lnlike_star += np.log(hists[i][0][np.digitize(mystar[i], hists[i][1])-1])
# lnlike_star -= n_nearby**5
# 
# 
# manual_ub = np.array([ 100., 50., 40., 8., 0., 2.])
# manual_lb = np.array([-100.,-70.,-30.,-6.,-7.,-7.])
# 
# man_mask = np.where(
#     np.all(
#         (gaia_xyzuvw < manual_ub) & (gaia_xyzuvw > manual_lb),
#         axis=1)
# )
# print(man_mask[0].shape)
# man_nearby_gaia = gaia_xyzuvw[man_mask]
# 
# 
# # plot histograms
# labels = 'XYZUVW'
# for i, label in enumerate(labels):
#     plt.clf()
#     plt.hist(man_nearby_gaia[:,i], bins=20)
#     plt.xlabel(label)
#     plt.savefig("temp_plots/man_nearby_gaia_{}.pdf".format(label))
# 
# man_hists = []
# bins = 20
# man_n_nearby = man_nearby_gaia.shape[0]
# # norm = n_nearby ** (5./6)
# for col in nearby_gaia.T:
#     man_hists.append(np.histogram(col, bins))
# #
# hist6d = np.histogramdd(man_nearby_gaia, bins=3)
# 
# # for nbins in range(10,40):
# #     all_hist6d = np.histogramdd(gaia_xyzuvw, bins=nbins)
# #     bin_widths = [bins[1] - bins[0] for bins in all_hist6d[1]]
# #     bin_area = np.prod(bin_widths)
# #
# #     bpmean = np.array([0,0,0,0,-4,-2])
# #     bp_ix = tuple([np.digitize(bpmean[dim],all_hist6d[1][dim]) - 1
# #                    for dim in range(6)])
# #     density_near_bp = all_hist6d[0][bp_ix] / bin_area
# #     print("{:02}: {:.5} | {}".format(nbins,density_near_bp), bin_widths)
# 
# for nbins in range(2,10):
#     near_hist6d = np.histogramdd(nearby_gaia, bins=nbins)
#     bin_widths = [bins[1] - bins[0] for bins in near_hist6d[1]]
#     bin_area = np.prod(bin_widths)
# 
#     bpmean = np.array([0,0,0,0,-4,-2])
#     bp_ix = tuple([np.digitize(bpmean[dim],near_hist6d[1][dim]) - 1
#                    for dim in range(6)])
#     density_near_bp = near_hist6d[0][bp_ix] / bin_area
#     print("{:02}: {:.5} | {}".format(nbins,density_near_bp, bin_widths))
# 
# # for dim, label in enumerate(labels):
# #     plt.clf()
# #     plt.bar(hist6d[1][dim],np.sum)
# #     plt.hist(man_nearby_gaia[:,dim], bins=20)
# #     plt.xlabel(label)
# #     plt.savefig("temp_plots/man_nearby_gaia_{}.pdf".format(label))
