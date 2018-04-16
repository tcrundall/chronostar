#! /usr/bin/env python
"""using the determinant of the star cov_matrices as a measure of occupied
volume, investigate if phase-space occupied by the stars remains constant
with traceback age or not
"""

import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys
sys.path.insert(0, '..')

import chronostar.groupfitter as gf

def get_dets(star_pars, time):
    """Calculates the determinants of the covariance matrices at a traceback age

    Parameters
    ----------
    xyzuvw_cov : [nstars, ntimes, 6, 6] array
    time : float [0.0, +infty)

    Returns
    -------
    dets : [nstars] array
    """
    interp_covs, _ = gf.interp_cov(target_time=time, star_pars=star_pars)
    dets = np.linalg.det(interp_covs)
    return dets

def lnprior_on_dets(star_pars, time):
    dets = get_dets(star_pars, time)
    lnprior = np.sum(-4 * np.log(dets))
    return lnprior


bp_tb_file = "../data/bp_TGAS2_traceback_save.pkl"
rave_twa_tb_file = "../data/tb_RAVE_twa_combined.pkl"
quad_tb_file = "tb_quad_data.pkl"

#tb_files = [bp_tb_file, rave_twa_tb_file]
tb_files = [quad_tb_file]

for tb_file in tb_files:
    star_pars = gf.read_stars(tb_file)
    nstars = int(star_pars['xyzuvw_cov'].shape[0])
    times = star_pars['times']
    ntimes = len(times)
    all_dets = np.zeros((nstars,ntimes))
    inverse_lnprior = np.zeros(len(times))
    for i, time in enumerate(times):
        all_dets[:,i] = get_dets(star_pars, time)
        inverse_lnprior[i] = - lnprior_on_dets(star_pars, time)

    max_dets = np.max(all_dets, axis=1)
    tiled_max = np.tile(max_dets, (ntimes, 1)).T

    prod_dets = np.prod(all_dets, axis=0)
    pdb.set_trace()

    normed_dets = all_dets / tiled_max
    normed_dets = normed_dets * 100000

    plt.plot(times, normed_dets.T)
    plt.xlabel("Times [Myr]")
    plt.ylabel("Normalised cov_matrix determinants")
    plt.title("'Effective volumes' with traceback age")

    plt.savefig(tb_file[:-4] + "T.png")

#    plt.clf()
#    plt.plot(all_dets)
#    plt.savefig("{}.png".format(tb_file[8:14]))
#    plt.clf()
#    plt.plot(all_dets.T)
#    plt.savefig("{}T.png".format(tb_file[:5]))
