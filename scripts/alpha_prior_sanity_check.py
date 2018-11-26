"""
This script explores the impact the prior on the virial should have on
permitting low density components mapping to the field.
"""
from __future__ import print_function, division

import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.synthesiser as syn
import chronostar.groupfitter as gf
import chronostar.expectmax as em

# 1 comp fit
one_group_pars_ex = [
    np.array([27.21857533, 40.84870565, 23.3006967 , -0.96968654, -3.47371966,
              -0.29458631, 16.70523603,  1.15738955, 14.07591161])
]
one_weights = [61.9987919]

# 2 comp fit
two_group_pars_ex = [
    np.array([25.24100057, 35.37151966, 23.73419406,  0.68106934, -3.77992686,
              -0.43692936,  8.97195291,  0.92658985, 13.33282707]),
    np.array([ 2.97783565e+01,  4.85637518e+01,  2.45364372e+01, -2.58377259e+00,
              -3.04098661e+00, -3.34909874e-02,  2.16584223e+01,  8.31520834e-01,
               1.53740845e+01])
]
two_weights = [33.02902374, 28.97088176,  6.0000945]

# 3 comp fit
three_group_pars_ex = [
    np.array([26.54427126, 37.99652339, 24.08375344,  0.36762423, -3.7307971,
              -0.34351942,  9.02614369,  0.92211963, 14.14381967]),
    np.array([  8.93722359,  22.28633376,  -1.18878485,  -8.63887102,
              -11.130962,  -3.43125315,   8.51810544,   1.77553,
                1.51311008]),
    np.array([ 3.21488735e+01,  4.88207224e+01,  2.43963555e+01, -2.75347262e+00,
              -2.98063139e+00,  1.26400066e-02,  2.21896254e+01,  7.69173481e-01,
               1.60651391e+01])
]
three_weights = [36.60656637,  4.70398271, 24.68926151,  2.00018941]

# overallLnLike
overallLnLikes = [-1092.54943463, -1063.00062202, -1010.87585206]
BICs = [2234.42221209, 2224.6479297, 2169.72173262]

# gather everything up
all_group_pars = [one_group_pars_ex,
                  two_group_pars_ex,
                  three_group_pars_ex]
all_weights = [one_weights,
               two_weights,
               three_weights]

# For each fit
for i, (group_pars_ex, weights) in enumerate(zip(all_group_pars, all_weights)):
    lnalpha_priors = []

    print("\n---  {} component fit  ---".format(len(group_pars_ex)))
    print("Likelihood from overlaps: {:8.4f}".format(overallLnLikes[i]))
    print("BIC: {:8.4f}".format(BICs[i]))
    for group_par, weight in zip(group_pars_ex, weights):
        group_obj = syn.Group(group_par, starcount=False, internal=False,
                              sphere=True)
        lnalpha_prior = gf.lnAlphaPrior(group_obj.getInternalSphericalPars(),
                                              weight)
        lnalpha_priors.append(lnalpha_prior)
        print("nstars: {:6.3f} | age: {:6.3f} | dX: {:6.3f} | dV: {:6.3f} |"
              "lnalpha_pr: {:6.3f}"\
              .format(weight, group_obj.age, group_obj.dx, group_obj.dv,
                      lnalpha_prior))
    #print(lnalpha_priors)


# for deeper insight, lets investigate the ratio of overlap between
# the flat bg field and the crappy 1.5 Myr component (3_comp[1]) and
# see if the difference will be corrected for by the large prior

rdir = "../results/em_fit/cf-15/"
star_pars_file = "../data/bpmg_cand_w_gaia_dr2_astrometry_comb_binars_xyzuvw.fits"

star_pars = gf.loadXYZUVW(star_pars_file)
bg_hists = np.load(rdir + "bg_hists.npy")
final_z = np.load(rdir + "final/final_membership.npy")
final_groups = np.load(rdir + "final_groups.npy")

for i in range(len(three_group_pars_ex)):
    spec_comp_stars_mask = np.where(final_z[:,i] > .5)
    spec_comp_star_pars = {'xyzuvw':star_pars['xyzuvw'][spec_comp_stars_mask],
                          'xyzuvw_cov':star_pars['xyzuvw_cov'][spec_comp_stars_mask]}
    spec_comp_group = syn.Group(three_group_pars_ex[1], starcount=False,
                               internal=False)

    spec_ln_bg_ols = em.backgroundLogOverlaps(spec_comp_star_pars['xyzuvw'], bg_hists,
                                         correction_factor=1.)

    # BUG IS FROM ME FORGETTING TO INCORPORATE AMPLITUDE OF
    # GROUP!!!
    weight = np.sum(final_z[:,i])
    print(weight)
    spec_ln_comp_ols = np.log(weight) + \
                       gf.getLogOverlaps(spec_comp_group.\
                                         getInternalSphericalPars(),
                                         spec_comp_star_pars)
    try:
        assert np.all(spec_ln_comp_ols > spec_ln_bg_ols)
    except AssertionError:
        print("Stars {}\n  are members of component {} despite"
              " having stronger overlap with background".\
              format(np.where(spec_ln_comp_ols<spec_ln_bg_ols), i))
        break


# good_comp_stars_mask = np.where(final_z[:,0] > .5)
# good_comp_star_pars = {'xyzuvw':star_pars['xyzuvw'][good_comp_stars_mask],
#                       'xyzuvw_cov':star_pars['xyzuvw_cov'][good_comp_stars_mask]}
# good_comp_group = syn.Group(three_group_pars_ex[0], starcount=False,
#                            internal=False)
# good_ln_bg_ols = em.backgroundLogOverlaps(good_comp_star_pars['xyzuvw'], bg_hists,
#                                      correction_factor=15.)
# good_ln_comp_ols = gf.getLogOverlaps(good_comp_group.getInternalSphericalPars(),
#                                 good_comp_star_pars)
#
#
# fine_comp_stars_mask = np.where(final_z[:,2] > .5)
# fine_comp_star_pars = {'xyzuvw':star_pars['xyzuvw'][fine_comp_stars_mask],
#                       'xyzuvw_cov':star_pars['xyzuvw_cov'][fine_comp_stars_mask]}
# fine_comp_group = syn.Group(three_group_pars_ex[0], starcount=False,
#                            internal=False)
# fine_ln_bg_ols = em.backgroundLogOverlaps(fine_comp_star_pars['xyzuvw'], bg_hists,
#                                      correction_factor=15.)
# fine_ln_comp_ols = gf.getLogOverlaps(fine_comp_group.getInternalSphericalPars(),
#                                 fine_comp_star_pars)
#
# #BUG!!!
#

