#! /usr/bin/env python
from __future__ import print_function, division

import chronostar.synthdata

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    can_plot = True
except ImportError:
    can_plot = False

from distutils.dir_util import mkpath
from distutils.errors import DistutilsFileError
import logging
import numpy as np
import sys
from emcee.utils import MPIPool
sys.path.insert(0, '..')
import chronostar.synthdata as syn
import chronostar.traceorbit as torb
import chronostar.retired2.converter as cv
import chronostar.fitplotter as fp
import chronostar.retired2.datatool as dt

try:
    run_name = sys.argv[1]
except:
    run_name = 'synth_bpmg'

try:
    rdir = "/data/mash/tcrun/em_fit/{}/".format(run_name)
    path_msg = "Storing data on mash data server"
    mkpath(rdir)
except (IOError, DistutilsFileError):
    path_msg = ("I'm guessing you're not Tim Crundall..."
                "or not on an RSAA server")
    rdir = "../results/em_fit/{}/".format(run_name)
    mkpath(rdir)

logging.basicConfig(
    level=logging.INFO,
    filename=rdir + 'em.log',
)

# Initialize the MPI-based pool used for parallelization.
using_mpi = True
try:
    pool = MPIPool()
    logging.info("Successfully initialised mpi pool")
except:
    #print("MPI doesn't seem to be installed... maybe install it?")
    logging.info("MPI doesn't seem to be installed... maybe install it?")
    using_mpi = False
    pool=None
if using_mpi:
    if not pool.is_master():
        print("One thread is going to sleep")
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)
print("Only one thread is master")

logging.info(path_msg)
print("Master should be working in the directory:\n{}".format(rdir))

# Setting up standard filenames
# Partial data generation saved in results/em_fit/[run_name]/synth_data/
sd_dir = rdir + 'synth_data/'
mkpath(sd_dir)
xyzuvw_perf_file     = sd_dir + 'perf_xyzuvw.npy'
groups_savefile      = sd_dir + 'origins.npy'
xyzuvw_init_savefile = sd_dir + 'xyzuvw_init.npy'
astro_savefile       = sd_dir + 'astro_table.txt'

# Final XYZUVW data file stored in chronostar/data/ to replicate
# treatment of real data
xyzuvw_conv_savefile = '../data/{}_xyzuvw.fits'.format(run_name)

# Calculate the initial parameters for each component that correspond
# to the current day mean of mean_now
logging.info("---------- Generating synthetic data...")
# Set a current-day location around which synth stars will end up

ERROR = 1.0
group_pars = np.array([
    [ 26.11, 38.20, 23.59,  0.34, -3.87, -0.36,  2.23, -0.05,  14.32, 35],
    [ 35.25, 52.58, 24.00, -2.80, -2.85,  0.25,  3.07, -0.27, 16.72, 25],
])
ngroups = group_pars.shape[0]

# Get background stars

# get typical background density

# nbgstars = 10
try:
    all_xyzuvw_now_perf = np.load(xyzuvw_perf_file)
    origins = dt.loadGroups(groups_savefile)
    star_pars = dt.loadXYZUVW(xyzuvw_conv_savefile)
    logging.info("Loaded synth data from previous run")
except IOError:
    all_xyzuvw_init = np.zeros((0,6))
    all_xyzuvw_now_perf = np.zeros((0,6))
    origins = []
    for i in range(ngroups):
        logging.info(" generating from group {}".format(i))
        # MANUALLY SEPARATE CURRENT DAY DISTROS IN DIMENSION X
        # mean_now_w_offset = mean_now.copy()
        # mean_now_w_offset[0] += i * 50
    
        # mean_then = torb.traceOrbitXYZUVW(mean_now_w_offset, -extra_pars[i,-2],
        #                                   single_age=True)
        xyzuvw_init, origin = syn.synthesiseXYZUVW(group_pars[i], form='sphere',
                                                   return_group=True,
                                                   internal=True)
        origins.append(origin)
        all_xyzuvw_init = np.vstack((all_xyzuvw_init, xyzuvw_init))
        xyzuvw_now_perf = torb.trace_many_cartesian_orbit(xyzuvw_init,
                                                          times=origin.age,
                                                          single_age=True)
        all_xyzuvw_now_perf = np.vstack((all_xyzuvw_now_perf, xyzuvw_now_perf))
    
    np.save(groups_savefile, origins)
    np.save(xyzuvw_perf_file, all_xyzuvw_now_perf)
    astro_table = chronostar.synthdata.measureXYZUVW(all_xyzuvw_now_perf, 1.0,
                                                     savefile=astro_savefile)
    star_pars = cv.convertMeasurementsToCartesian(
        astro_table, savefile=xyzuvw_conv_savefile,
    )

# make sure stars are initialised as expected
if can_plot:
    for dim1, dim2 in ('xy', 'xu', 'yv', 'zw', 'uv'):
        plt.clf()
        fp.plotPaneWithHists(dim1,dim2,groups=origins,
                             weights=[origin.nstars for origin in origins],
                             star_pars=star_pars,
                             group_now=True)
        plt.savefig(rdir + 'pre_plot_{}{}.pdf'.format(dim1, dim2))

# MAX_COMP = 5
# ncomps = 1
# # prev_groups = None
# # prev_meds = None
# prev_lnpost = -np.inf
# prev_BIC = np.inf
# prev_lnlike = -np.inf
#
# while ncomps < MAX_COMP:
#     # handle special case of one component
#     if ncomps == 1:
#         logging.info("******************************************")
#         logging.info("*********  FITTING 1 COMPONENT  **********")
#         logging.info("******************************************")
#         run_dir = rdir + '{}/'.format(ncomps)
#         mkpath(run_dir)
#
#         try:
#             new_groups = dt.loadGroups(run_dir + 'final/final_groups.npy')
#             new_meds = np.load(run_dir + 'final/final_med_errs.npy')
#             new_z = np.load(run_dir + 'final/final_membership.npy')
#             logging.info("Loaded from previous run")
#         except IOError:
#             new_groups, new_meds, new_z =\
#                 em.fitManyGroups(star_pars, ncomps, rdir=run_dir, pool=pool)
#             new_groups = np.array(new_groups)
#
#         new_lnlike = em.getOverallLnLikelihood(star_pars, new_groups,
#                                                bg_ln_ols=None)
#         new_lnpost = em.getOverallLnLikelihood(star_pars, new_groups,
#                                                bg_ln_ols=None,
#                                                inc_posterior=True)
#         new_BIC = em.calcBIC(star_pars, ncomps, new_lnlike)
#     # handle multiple components
#     else:
#         logging.info("******************************************")
#         logging.info("*********  FITTING {} COMPONENTS  *********".\
#                      format(ncomps))
#         logging.info("******************************************")
#         best_fits = []
#         lnlikes = []
#         lnposts = []
#         BICs = []
#         all_meds = []
#         all_zs = []
#
#         # iteratively try subdividing each previous group
#         for i, split_group in enumerate(prev_groups):
#             logging.info("-------- decomposing {}th component ---------".\
#                          format(i))
#             run_dir = rdir + '{}/{}/'.format(ncomps, chr(ord('A') + i))
#             mkpath(run_dir)
#
#             # Decompose and replace the ith group with two new groups
#             # by using the 16th and 84th percentile ages from chain
#             _, split_groups = em.decomposeGroup(split_group,
#                                                 young_age = prev_meds[i,-1,1],
#                                                 old_age = prev_meds[i,-1,2])
#             init_groups = list(prev_groups)
#             init_groups.pop(i)
#             init_groups.insert(i, split_groups[1])
#             init_groups.insert(i, split_groups[0])
#
#             # run em fit
#             try:
#                 groups = dt.loadGroups(run_dir + 'final/final_groups.npy')
#                 meds = np.load(run_dir + 'final/final_med_errs.npy')
#                 z = np.load(run_dir + 'final/final_membership.npy')
#                 logging.info("Fit loaded from previous run")
#             except IOError:
#                 groups, meds, z = \
#                     em.fitManyGroups(star_pars, ncomps, rdir=run_dir, pool=pool,
#                                      init_groups=init_groups)
#             best_fits.append(groups)
#             all_meds.append(meds)
#             all_zs.append(z)
#
#             lnlike = em.getOverallLnLikelihood(star_pars, groups,
#                                                bg_ln_ols=None)
#             lnlikes.append(lnlike)
#             lnposts.append(em.getOverallLnLikelihood(star_pars, groups,
#                                                      bg_ln_ols=None,
#                                                      inc_posterior=True))
#             BICs.append(em.calcBIC(star_pars, ncomps, lnlike))
#             logging.info("Decomp finished with\nBIC: {}\nLnlike: {}".format(
#                 BICs[-1], lnlikes[-1]
#             ))
#
#         # identify the best performing decomposition
#         best_split_ix = np.argmax(lnposts)
#         new_groups, new_meds, new_z, new_lnlike, new_lnpost, new_BIC = \
#             zip(best_fits, all_meds, all_zs,
#                 lnlikes, lnposts, BICs)[best_split_ix]
#         logging.info("Selected {} as best decomposition".format(i))
#         logging.info("Turned\n{}".format(
#             prev_groups[best_split_ix].getInternalSphericalPars()))
#         logging.info("into\n{}\n&\n{}".format(
#             new_groups[best_split_ix].getInternalSphericalPars(),
#             new_groups[best_split_ix+1].getInternalSphericalPars(),
#         ))
#
#     # Check if the fit has improved
#     if new_BIC < prev_BIC:
#         logging.info("Extra component has improved BIC...")
#         logging.info("New BIC: {} < Old BIC: {}".format(new_BIC, prev_BIC))
#         logging.info("lnlike: {} | {}".format(new_lnlike, prev_lnlike))
#         logging.info("lnpost: {} | {}".format(new_lnpost, prev_lnpost))
#         prev_groups, prev_meds, prev_z, prev_lnlike, prev_lnpost, \
#         prev_BIC = \
#             (new_groups, new_meds, new_z, new_lnlike, new_lnpost, new_BIC)
#         ncomps += 1
#     else:
#         logging.info("Extra component has worsened BIC...")
#         logging.info("New BIC: {} < Old BIC: {}".format(new_BIC, prev_BIC))
#         logging.info("lnlike: {} | {}".format(new_lnlike, prev_lnlike))
#         logging.info("lnpost: {} | {}".format(new_lnpost, prev_lnpost))
#         break
#
#     logging.info("Best fit:\n{}".format(
#         [group.getInternalSphericalPars() for group in prev_groups]))
#
# if using_mpi:
#     pool.close()
