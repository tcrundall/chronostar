from __future__ import print_function, division

"""
For investigating accuracy of field distribution required to provide
realistic BMPG membership probabilities based on current best fit to group
"""

import logging
import numpy as np
import matplotlib as pyplot
from distutils.dir_util import mkpath
import sys

from astropy.io import fits

sys.path.insert(0, '..')
import chronostar.synthesiser as syn
import chronostar.traceorbit as torb
import chronostar.transform as tf
import chronostar.datatool as dt

def MVGaussian(vec_x, mean, cov, inv_cov = None):
    """
    Evaluate the MVGaussian defined by mean and cov at vec_x

    Parameters
    ----------
    vec_x : [dim] float array
        the point at which to evaluate the function
    mean : [dim] float array
        the mean of the MVGaussian distribution
    cov : [dim, dim] float array
        the covaraince matrix of the MVGaussian distribution

    Returns
    -------
    (float)
        evaluation of vec_x
    """
    if inv_cov is None:
        inv_cov = np.linalg.inv(cov)
    dim = vec_x.shape[0]

    assert (mean.shape == vec_x.shape)
    assert (cov.shape == (dim, dim))
    coeff = 1./np.sqrt((2*np.pi)**dim * np.linalg.det(cov))

    diff = vec_x - mean
    expon = -0.5 * np.dot(diff,
                          np.dot(inv_cov, diff))
    return coeff * np.exp(expon)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

rdir = "../results/em_fit/gaia_dr2_bp/"

# final groups represent the mode of the final sampling stage
final_groups_file = rdir + "final/final_groups.npy"
final_chain0_file = rdir + "final/group0/final_chain.npy"
final_chain1_file = rdir + "final/group1/final_chain.npy"
final_membership_file = rdir + "final/final_membership.npy"
bp_xyzuvw_file = "../data/gaia_dr2_bp_xyzuvw.fits"
gaia_xyzuvw_mean_file = "../data/gaia_dr2_mean_xyzuvw.npy"
gaia_astr_file = "../data/all_rvs_w_ok_plx.fits"

# analysis files
andir = "../results/bp_members/"
mkpath(andir)
gaia_bpmg_evals_file = andir + "gaia_bpmg_evals.npy"
gaia_gaia_evals_file = andir + "gaia_gaia_evals.npy"
bpmg_candidates_mask_file = andir + "bpmg_candidates_mask.npy"
gaia_mean_file = andir + "gaia_mean.npy"
gaia_cov_file = andir + "gaia_cov.npy"
bpmg_memb_probs_file = andir + "bpmg_memb_probs.npy"

gaia_xyzuvw = np.load(gaia_xyzuvw_mean_file)
z_final = np.load(final_membership_file)
bp_hdul = fits.open(bp_xyzuvw_file)
gaia_hdul = fits.open(gaia_astr_file)
bp_xyzuvw = bp_hdul[1].data
bp_xyzuvw_cov = bp_hdul[2].data

bp_core_mask = np.where(z_final[:,0] > 0.75)
bp_core_xyzuvw = bp_xyzuvw[bp_core_mask]


bpmg_then = np.load(final_groups_file)[0]
bpmg_mean_now = torb.traceOrbitXYZUVW(bpmg_then.mean, bpmg_then.age,
                                      single_age=True)
bpmg_cov_now = tf.transformCovMat(
    bpmg_then.generateCovMatrix(), torb.traceOrbitXYZUVW,
    bpmg_then.mean, dim=6,
    args=(bpmg_then.age, True)
)

ngaia_stars = gaia_xyzuvw.shape[0]

try:
    gaia_bpmg_evals = np.load(gaia_bpmg_evals_file)
except IOError:
    print("Evaluating gaia stars at BPMG current MVGauss distribution")
    gaia_bpmg_evals = np.zeros(ngaia_stars)
    bpmg_invcov_now = np.linalg.inv(bpmg_cov_now)
    for i, gaia_star in enumerate(gaia_xyzuvw):
        if (i % 100000 == 0):
            print("{:10} of {:10}... {:6.1f}%".format(i, ngaia_stars,
                                                      i / ngaia_stars*100))
        # UNTESTED!!!
        gaia_bpmg_evals[i] = MVGaussian(gaia_star, bpmg_mean_now,
                                        bpmg_cov_now,
                                        inv_cov=bpmg_invcov_now)
    np.save(gaia_bpmg_evals_file, gaia_bpmg_evals)

bpmg_candidates_mask = np.where(gaia_bpmg_evals >
                                np.percentile(gaia_bpmg_evals,99.99))
np.save(bpmg_candidates_mask_file, bpmg_candidates_mask)

try:
    gaia_mean = np.load(gaia_mean_file)
    gaia_cov = np.load(gaia_cov_file)
except IOError:
    print("Evaluating gaia mean and cov")
    gaia_mean = np.mean(gaia_xyzuvw, axis=0)
    gaia_cov_sum = np.zeros((6,6))
    for i, gaia_star in enumerate(gaia_xyzuvw):
        if (i % 100000 == 0):
            print("{:10} of {:10}... {:6.1f}%".format(i, ngaia_stars,
                                               i/ngaia_stars*100))
        diff = gaia_star - gaia_mean
        gaia_cov_sum += np.outer(diff, diff)
    gaia_cov = gaia_cov_sum / ngaia_stars
    np.save(gaia_mean_file, gaia_mean)
    np.save(gaia_cov_file, gaia_cov)


try:
    gaia_gaia_evals = np.load(gaia_gaia_evals_file)
except IOError:
    print("Evaluating gaia stars at gaia single MVGauss distribution")
    gaia_gaia_evals = np.zeros(ngaia_stars)
    gaia_invcov = np.linalg.inv(gaia_cov)
    for i, gaia_star in enumerate(gaia_xyzuvw):
        if (i % 100000 == 0):
            print("{:10} of {:10}... {:6.1f}%".format(i, ngaia_stars,
                                               i/ngaia_stars*100))
        # UNTESTED!!!
        gaia_gaia_evals[i] = MVGaussian(gaia_star, gaia_mean,
                                        gaia_cov, inv_cov=gaia_invcov)
    np.save(gaia_gaia_evals_file, gaia_gaia_evals)

nbpmg_stars = 100
print("Beginning iteration: n = {:5.4f}".format(nbpmg_stars))
for i in range(6):
    memb_probs = (nbpmg_stars * gaia_bpmg_evals /
                  (nbpmg_stars * gaia_bpmg_evals +
                   ( (ngaia_stars-nbpmg_stars) * gaia_gaia_evals) ))
    nbpmg_stars = np.sum(memb_probs)
    print("Iteration {}: n = {:5.4f}".format(i, nbpmg_stars))

np.save(bpmg_memb_probs_file, memb_probs)
best_bpmg_mask = np.where(memb_probs > 0.7)
all_bpmg_mask = np.where(memb_probs > 0.0001)

mkpath("../results/gaia_dr2_bp_best/")
best_bpmg_astro_file = "../data/gaia_dr2_bp_best_astro.fits"
dt.createSubFitsFile(best_bpmg_mask, best_bpmg_astro_file)
dt.convertGaiaToXYZUVWDict(best_bpmg_astro_file)
np.save("../results/gaia_dr2_bp_best/init_z.npy",
        memb_probs[best_bpmg_mask])

mkpath("../results/gaia_dr2_bp_all_poss/")
all_poss_bpmg_astro_file = "../data/gaia_dr2_bp_all_poss_astro.fits"
dt.createSubFitsFile(all_bpmg_mask, all_poss_bpmg_astro_file)
dt.convertGaiaToXYZUVWDict(all_poss_bpmg_astro_file)
np.save("../results/gaia_dr2_bp_all_poss/init_z.npy",
        memb_probs[all_bpmg_mask])




