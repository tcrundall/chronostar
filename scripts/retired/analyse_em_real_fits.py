from __future__ import division, print_function

"""
Use this script to analyse the results of a multi-component fit 
(with a background) to real data
"""

import logging
import numpy as np
import os
import sys
sys.path.insert(0, '..')

import chronostar.compfitter as gf
import chronostar.expectmax as em
import chronostar.retired.hexplotter as hp
import chronostar.traceorbit as torb
import chronostar.transform as tf

def calcBIC(star_pars, ncomps, lnlike):
    nstars = len(star_pars['xyzuvw'])
    n = nstars * 7 # 6 for phase space origin and 1 for age
    k = ncomps * 8 # 6 for central estimate, 2 for dx and dv, 1 for age
    return np.log(n)*k - 2 * lnlike

logging.basicConfig(level=logging.INFO, filemode='w',
                    filename='temp_logs/analysing_em.log')

assoc_name = "bpmg_cand_w_gaia_dr2_astrometry_comb_binars"
rdir_stem = "../results/em_fit/"

if not os.path.exists(rdir_stem):  # on server
    rdir_stem = "/data/mash/tcrun/em_fit/"
    
for ncomps in range(1,4):
    try:
        rdir = rdir_stem + assoc_name + "_{}/".format(ncomps)
        ddir = "../data/"


        # final_z_file = rdir + "final/final_membership.npy"
        final_z_file = rdir + "memberships.npy"
        final_groups_file = rdir + "final_groups.npy"
        bg_hists_file = rdir + "bg_hists.npy"
        data_file = ddir + assoc_name + "_xyzuvw.fits"
        star_pars = gf.loadXYZUVW(data_file)

        z = np.load(final_z_file)
        # import pdb; pdb.set_trace()
        groups = np.load(final_groups_file)
        bg_hists = np.load(bg_hists_file)

        bg_ln_ols = em.backgroundLogOverlaps(star_pars['xyzuvw'], bg_hists)

        overall_lnlike, z = em.getOverallLnLikelihood(star_pars, groups,
                                                   bg_ln_ols, return_z=True)
        print("overall_lnlike with {} comps is: {:.5}".format(ncomps, overall_lnlike))
        bic = calcBIC(star_pars, ncomps, overall_lnlike)
        print("BIC is: {}".format(bic))
        print("With {:.2} stars accounted for by background"\
              .format(np.sum(z[:,-1])))

        means = {}
        means['fitted_then'] = [g.mean for g in groups]
        means['fitted_now'] = [
            torb.trace_cartesian_orbit(g.mean, g.age, single_age=True)
            for g in groups
        ]
        covs = {}
        covs['fitted_then'] = np.array([
            g.generateSphericalCovMatrix() for g in groups
        ])
        covs['fitted_now'] = np.array([
                tf.transform_covmatrix(covs['fitted_then'][0], torb.trace_cartesian_orbit,
                                       means['fitted_then'][0],
                                       args=(g.age,True)
                                       )
            for g in groups
            ])
        hp.plotNewQuad(star_pars, means, covs, None, 'final',
                       save_dir='temp_plots/',
                       file_stem='XY-UV_ncomps_{}_'.format(ncomps), z=z,
                       title="BIC: {}, likelihood: {}".format(bic, overall_lnlike),
                       top_dims=(0,1), bot_dims=(3,4))

        hp.plotNewQuad(star_pars, means, covs, None, 'final',
                       save_dir='temp_plots/',
                       file_stem='XU-YV_ncomps_{}_'.format(ncomps), z=z,
                       title="BIC: {}, likelihood: {}".format(bic, overall_lnlike),
                       top_dims=(0,3), bot_dims=(1,4))

        hp.plotNewQuad(star_pars, means, covs, None, 'final',
                       save_dir='temp_plots/',
                       file_stem='ZX-ZW_ncomps_{}_'.format(ncomps), z=z,
                       title="BIC: {}, likelihood: {}".format(bic, overall_lnlike),
                       top_dims=(2,1), bot_dims=(2,5))

    except IOError:
        print("Missing results for ncomps={}".format(ncomps))
