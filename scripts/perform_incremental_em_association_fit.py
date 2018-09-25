#! /usr/bin/env python

"""
Fit a single component to association.
Decompose single component into two, and fit.
If BIC (or some other measure) is improved, accept new 2-part
decopmosition and continue:
Decompose each of the two components into two, one at a time, and fit.
Pick the best one by posterior.
If BIC (or some other measure) is improved, accept new 3-part
decomposition and continue:
"""

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    can_plot = True
except ImportError:
    can_plot = False

from distutils.dir_util import mkpath
from distutils.errors import DistutilsFileError
import os
import logging
import numpy as np
import pdb
import sys
from emcee.utils import MPIPool
sys.path.insert(0, '..')
import chronostar.expectmax as em
import chronostar.datatool as dt

try:
    ass_name = sys.argv[1]
except IndexError:
    print(" ---------------- INCORRECT USAGE ---------------\n"
          "  Usage: nohup mpirun -np 19 python\n"
          "         perform_em_association_fit.py [ass_name]\n"
          " ------------------------------------------------")
    print("Using bpmg as default...")
    ass_name = 'bpmg_cand_w_gaia_dr2_astrometry_comb_binars'

# Setting key parameters for fit
CORRECTION_FACTOR = 15.     # maybe don't even need this...

if os.path.isdir('/data/mash/tcrun/'):
    rdir = "/data/mash/tcrun/em_fit/{}/".format(ass_name)

    gdir = "/data/mash/tcrun/" # directory with master gaia data
    path_msg = "Storing data on mash data server"
    mkpath(rdir)
else:
    rdir = "../results/em_fit/{}/".format(ass_name)
    gdir = "../data/" # directory with master gaia data
    path_msg = "Storing data on mash data server"
    mkpath(rdir)

gaia_xyzuvw_file = gdir + 'gaia_dr2_mean_xyzuvw.npy'
xyzuvw_file = '../data/' + ass_name + '_xyzuvw.fits'
best_fit_file = rdir + "best_fit.npy"
bg_hist_file = rdir + "bg_hists.npy"

logging.basicConfig(
    level=logging.INFO, filemode='w',
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

#logging.info(path_msg)

print("Master should be working in the directory:\n{}".format(rdir))

star_pars = dt.loadXYZUVW(xyzuvw_file)

# -------------------------------------------------------------
# constract histograms of Gaia stars in vicinity of association
# -------------------------------------------------------------
logging.info("Building histograms of Gaia stars in vicinity of associaiton")
star_means = star_pars['xyzuvw']
margin = 2.

# construct box
kin_max = np.max(star_means, axis=0)
kin_min = np.min(star_means, axis=0)
span = kin_max - kin_min
upper_boundary = kin_max + margin*span
lower_boundary = kin_min - margin*span

# get gaia stars within box
gaia_xyzuvw = np.load(gaia_xyzuvw_file)
mask = np.where(
    np.all(
        (gaia_xyzuvw < upper_boundary) & (gaia_xyzuvw > lower_boundary),
        axis=1)
)
nearby_gaia = gaia_xyzuvw[mask]
bg_hists = []
bins = int(margin**1.5 * 25)
n_nearby = nearby_gaia.shape[0]
norm = n_nearby ** (5./6)
for col in nearby_gaia.T:
    bg_hists.append(np.histogram(col, bins))
np.save(rdir+bg_hist_file, bg_hists)
logging.info("Histograms constructed with {} stars, stored in {}".format(
    n_nearby, rdir+bg_hist_file
))

# --------------------------------------------------------------------------
# Get initial parameters
# --------------------------------------------------------------------------
# init_origin = False
# origins = None
# if NGROUPS == 1:
#     init_origin = True
#     nstars = star_means.shape[0]
#     bp_mean = np.mean(star_means, axis=0)
#     bp_cov = np.cov(star_means.T)
#     bp_dx = np.sqrt(np.min([bp_cov[0,0], bp_cov[1,1], bp_cov[2,2]]))
#     bp_dv = np.sqrt(np.min([bp_cov[3,3], bp_cov[4,4], bp_cov[5,5]]))
#     bp_age = 0.5
#     bp_pars = np.hstack((bp_mean, bp_dx, bp_dv, bp_age, nstars))
#     bp_group = syn.Group(bp_pars)
#     origins = [bp_group]

#go through and compare overlap with groups with
#background overlap
#
#bp_pars = np.array([
#   2.98170398e+01,  4.43573995e+01,  2.29251498e+01, -9.65731744e-01,
#   -3.42827894e+00, -3.99928052e-02 , 2.63084094e+00,  1.05302890e-01,
#   1.59367119e+01, nstars
#])
#bp_group = syn.Group(bp_pars)
#
# --------------------------------------------------------------------------
# Run fit
# --------------------------------------------------------------------------
# logging.info("Everythign loaded, about to fit with {} components"\
#     .format(NGROUPS))

# !!! plotPaneWithHists not set up to handle None groups and None weights
# # make sure stars are initialised as expected
# if can_plot:
#     for dim1, dim2 in ('xy', 'xu', 'yv', 'zw', 'uv'):
#         plt.clf()
#         fp.plotPaneWithHists(dim1, dim2, star_pars=star_pars)
#         plt.savefig(rdir + 'pre_plot_{}{}.pdf'.format(dim1, dim2))

logging.info("Using data file {}".format(xyzuvw_file))
MAX_COMP = 5
ncomps = 1
# prev_groups = None
# prev_meds = None
prev_lnpost = -np.inf
prev_BIC = np.inf

while ncomps < MAX_COMP:
    # handle special case of one component
    if ncomps == 1:
        logging.info("******************************************")
        logging.info("*********  FITTING 1 COMPONENT  **********")
        logging.info("******************************************")
        run_dir = rdir + '{}/'.format(ncomps)
        mkpath(run_dir)

        try:
            new_groups = dt.loadGroups(run_dir + 'final/final_groups.npy')
            new_meds = np.load(run_dir + 'final/final_med_errs.npy')
            new_z = np.load(run_dir + 'final/final_membership.npy')
            logging.info("Loaded from previous run")
        except IOError:
            new_groups, new_meds, new_z = \
                em.fitManyGroups(star_pars, ncomps, rdir=run_dir, pool=pool,
                                 bg_hist_file=bg_hist_file)
            new_groups = np.array(new_groups)

        new_lnlike = em.getOverallLnLikelihood(star_pars, new_groups,
                                               bg_ln_ols=None)
        new_lnpost = em.getOverallLnLikelihood(star_pars, new_groups,
                                               bg_ln_ols=None,
                                               inc_posterior=True)
        new_BIC = em.calcBIC(star_pars, ncomps, new_lnlike)
    # handle multiple components
    else:
        logging.info("******************************************")
        logging.info("*********  FITTING {} COMPONENTS  *********". \
                     format(ncomps))
        logging.info("******************************************")
        best_fits = []
        lnlikes = []
        lnposts = []
        BICs = []
        all_meds = []
        all_zs = []

        # iteratively try subdividing each previous group
        for i, split_group in enumerate(prev_groups):
            logging.info("-------- decomposing {}th component ---------". \
                         format(i))
            run_dir = rdir + '{}/{}/'.format(ncomps, chr(ord('A') + i))
            mkpath(run_dir)

            # Decompose and replace the ith group with two new groups
            # by using the 16th and 84th percentile ages from chain
            _, split_groups = em.decomposeGroup(split_group,
                                                young_age=prev_meds[
                                                    i, -1, 1],
                                                old_age=prev_meds[i, -1, 2])
            init_groups = list(prev_groups)
            init_groups.pop(i)
            init_groups.insert(i, split_groups[1])
            init_groups.insert(i, split_groups[0])

            # run em fit
            try:
                groups = dt.loadGroups(run_dir + 'final/final_groups.npy')
                meds = np.load(run_dir + 'final/final_med_errs.npy')
                z = np.load(run_dir + 'final/final_membership.npy')
                logging.info("Fit loaded from previous run")
            except IOError:
                groups, meds, z = \
                    em.fitManyGroups(star_pars, ncomps, rdir=run_dir,
                                     pool=pool,
                                     init_groups=init_groups,
                                     bg_hist_file=bg_hist_file)
            best_fits.append(groups)
            all_meds.append(meds)
            all_zs.append(z)

            lnlike = em.getOverallLnLikelihood(star_pars, groups,
                                               bg_ln_ols=None)
            lnlikes.append(lnlike)
            lnposts.append(em.getOverallLnLikelihood(star_pars, groups,
                                                     bg_ln_ols=None,
                                                     inc_posterior=True))
            BICs.append(em.calcBIC(star_pars, ncomps, lnlike))
            logging.info("Decomp finished with\nBIC: {}\nLnlike: {}".format(
                BICs[-1], lnlikes[-1]
            ))

        # identify the best performing decomposition
        best_split_ix = np.argmax(lnposts)
        new_groups, new_meds, new_z, new_lnlike, new_lnpost, new_BIC = \
            zip(best_fits, all_meds, all_zs,
                lnlikes, lnposts, BICs)[best_split_ix]
        logging.info("Selected {} as best decomposition".format(i))
        logging.info("Turned\n{}".format(
            prev_groups[best_split_ix].getInternalSphericalPars()))
        logging.info("into\n{}\n&\n{}".format(
            new_groups[best_split_ix].getInternalSphericalPars(),
            new_groups[best_split_ix + 1].getInternalSphericalPars(),
        ))

    # Check if the fit has improved
    if new_lnpost > prev_lnpost:
        logging.info("Extra component has improved lnpost...")
        logging.info("{} > {}".format(new_lnpost, prev_lnpost))
        logging.info("New BIC: {} | Old BIC: {}".format(new_BIC, prev_BIC))
        prev_groups, prev_meds, prev_z, prev_lnlike, prev_lnpost, \
        prev_BIC = \
            (new_groups, new_meds, new_z, new_lnlike, new_lnpost, new_BIC)
        ncomps += 1
    else:
        logging.info("Extra component has worsened lnpost...")
        break

    logging.info("Best fit:\n{}".format(
        [group.getInternalSphericalPars() for group in prev_groups]))

if using_mpi:
    pool.close()
