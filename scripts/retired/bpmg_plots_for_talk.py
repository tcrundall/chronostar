from __future__ import division, print_function
"""
Generate some BPMG plots for talk
"""

import sys
sys.path.insert(0, "..")

import chronostar.retired.hexplotter as hp
import chronostar.compfitter as gf

xyzuvw_file = "../data/gaia_dr2_bp_xyzuvw.fits"
rdir = "../results/em_fit/gaia_dr2_bp/iter11/"

star_pars = gf.loadXYZUVW(xyzuvw_file)
#final_z = np.load(rdir + "final_groups.npy")
#final_med_errs = np.load(rdir + "final_med_errs.npy")
#final_groups = np.load(rdir + "final_groups.npy")

hp.dataGathererEM(2, 10, rdir, rdir, xyzuvw_file=xyzuvw_file)


