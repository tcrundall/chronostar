"""
Initial rough attempt of checking membership of a single star
"""

from astropy.table import Table
import numpy as np
import sys

import chronostar.likelihood

sys.path.insert(0, '..')

import chronostar.retired2.datatool as dt
import chronostar.retired2.converter as cv
import chronostar.coordinate as cc
import chronostar.expectmax as em

filename = '../data/2M0249-05.fits'
star_tab = Table.read(filename)
final_groups = '../results/em_fit/beta_Pictoris_wgs_inv2_5B_res/' \
               'final_groups.npy'
beta_fit = dt.loadGroups(final_groups)[0]
gaia_xyzuvw_file = '../data/gaia_dr2_mean_xyzuvw.npy'

if np.isnan(star_tab['radial_velocity']):
    print("Its nan")
    # from Shkolnik 2017
    # star_tab['radial_velocity'] = 14.4
    # star_tab['radial_velocity_error'] = 0.4

    star_tab['radial_velocity'] = 16.44
    star_tab['radial_velocity_error'] = 1.

if np.isnan(star_tab['radial_velocity']):
    print("Its nan")

# extend proper motion uncertainty
star_tab['pmra_error'] *= 100.
star_tab['pmdec_error'] *= 100.
# star_tab['radial_velocity_error'] *= 10

astr_mean, astr_cov = dt.convertRecToArray(star_tab[0])

xyzuvw_cov = cv.transformAstrCovsToCartesian(np.array([astr_cov]),
                                             np.array([astr_mean])
                                             )[0]
xyzuvw = cc.convert_astrometry2lsrxyzuvw(astr_mean)

ln_bg_ols = dt.getKernelDensities(gaia_xyzuvw_file, [xyzuvw])

star_pars = {'xyzuvw':np.array([xyzuvw]),
             'xyzuvw_cov':np.array([xyzuvw_cov])}
nbp_stars = 100
ln_bp_ols = np.log(nbp_stars) + chronostar.likelihood.get_lnoverlaps(beta_fit.getInternalSphericalPars(), star_pars)

combined_lnols = np.hstack((ln_bp_ols, ln_bg_ols))

membership_probs = em.calc_membership_probs(combined_lnols)
print(membership_probs)

