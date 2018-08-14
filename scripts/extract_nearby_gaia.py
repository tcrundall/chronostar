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

# fit gaus and flat line to distros
