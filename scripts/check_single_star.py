"""
Initial rough attempt of checking membership of a single star
"""

from astropy.table import Table
import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.datatool as dt
import chronostar.converter as cv
import chronostar.coordinate as cc
import chronostar.groupfitter as gf

filename = '../data/2M0249-05.fits'
star_tab = Table.read(filename)
final_groups = '../results/em_fit/beta_Pictoris_wgs_inv2_5B_res/' \
               'final_groups.npy'
beta_fit = dt.loadGroups(final_groups)[0]
gaia_xyzuvw_file = '../data/gaia_dr2_mean_xyzuvw.npy'

if np.isnan(star_tab['radial_velocity']):
    print("Its nan")
    star_tab['radial_velocity'] = 50.
    star_tab['radial_velocity_error'] = 100.

if np.isnan(star_tab['radial_velocity']):
    print("Its nan")

astr_mean, astr_cov = dt.convertRecToArray(star_tab[0])

xyzuvw_cov = cv.transformAstrCovsToCartesian(np.array([astr_cov]),
                                             np.array([astr_mean])
                                             )[0]
xyzuvw = cc.convertAstrometryToLSRXYZUVW(astr_mean)

ln_bg_ols = dt.getKernelDensities(gaia_xyzuvw_file, [xyzuvw])

star_pars = {'xyzuvw':np.array([xyzuvw]),
             'xyzuvw_cov':np.array([xyzuvw_cov])}

ln_bp_ols = gf.getLogOverlaps(beta_fit.getInternalSphericalPars(), star_pars)

print(ln_bg_ols)
print(ln_bp_ols)
