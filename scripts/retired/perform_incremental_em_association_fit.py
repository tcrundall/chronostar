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
import os
import logging
import numpy as np
import sys
from emcee.utils import MPIPool
sys.path.insert(0, '..')
import chronostar.expectmax as em
import chronostar.retired2.datatool as dt

try:
    ass_name = sys.argv[1]
except IndexError:
    print(" ---------------- INCORRECT USAGE ---------------\n"
          "  Usage: nohup mpirun -np 19 python\n"
          "         perform_em_association_fit.py [ass_name] [label]\n"
          " ------------------------------------------------")
    print("Using bpmg as default...")
    ass_name = 'bpmg_cand_w_gaia_dr2_astrometry_comb_binars'

# can optionally append a little label at the end
try:
    label = sys.argv[2]
    print("Read in a label!")
except IndexError:
    label = None

# Setting key parameters for fit
# CORRECTION_FACTOR = 15.     # maybe don't even need this...

if os.access('/data/mash/tcrun/', os.W_OK):
    rdir = "/data/mash/tcrun/em_fit/{}".format(ass_name.replace(' ','_'))
    # append optional label if provided
    if label:
        rdir += '_{}/'.format(label)
    else:
        rdir += '/'

    gdir = "/data/mash/tcrun/" # directory with master gaia data
    path_msg = "Storing data on mash data server"
    mkpath(rdir)
else:
    rdir = "../results/em_fit/{}".format(ass_name.replace(' ','_'))
    # append optional label if provided
    if label:
        rdir += '_{}/'.format(label)
    else:
        rdir += '/'
    gdir = "../data/" # directory with master gaia data
    path_msg = "Storing data on mash data server"
    mkpath(rdir)

gaia_xyzuvw_file = gdir + 'gaia_dr2_mean_xyzuvw.npy'
xyzuvw_file = '../data/' + ass_name + '_xyzuvw.fits'
bg_ln_ols_file = rdir + 'bg_ln_ols.npy'

logging.basicConfig(
    level=logging.INFO, filemode='a',
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

# GET BACKGROUND LOG OVERLAP DENSITIES:
logging.info("Acquiring background overlaps")
try:
    print("In try")
    logging.info("Calculating background overlaps")
    logging.info(" -- this step employs scipy's kernel density estimator")
    logging.info(" -- so could take a few minutes...")
    bg_ln_ols = np.load(bg_ln_ols_file)
    print("could load")
    assert len(bg_ln_ols) == len(star_pars['xyzuvw'])
    logging.info("Loaded bg_ln_ols from file")
except (IOError, AssertionError):
    print("in except")
    bg_ln_ols = dt.getKernelDensities(gaia_xyzuvw_file, star_pars['xyzuvw'], amp_scale=0.001)
    np.save(bg_ln_ols_file, bg_ln_ols)
print("continuing")
logging.info("DONE!")

# # !!! plotPaneWithHists not set up to handle None groups and None weights
# # make sure stars are initialised as expected
# if can_plot:
#     for dim1, dim2 in ('xy', 'xu', 'yv', 'zw', 'uv'):
#         plt.clf()
#         fp.plotPaneWithHists(dim1, dim2, star_pars=star_pars)
#         plt.savefig(rdir + 'pre_plot_{}{}.pdf'.format(dim1, dim2))

logging.info("Using data file {}".format(xyzuvw_file))
MAX_COMP = 10
ncomps = 1 # 5

# init_groups = [
#   group1, group2, group3, group4, group5,
# ]

# Set up initial values of results
prev_groups = None
prev_meds = None
prev_lnpost = -np.inf
prev_BIC = np.inf
prev_lnlike = -np.inf
prev_z = None

while ncomps < MAX_COMP:
    if ncomps >= MAX_COMP:
        logging.info("++++++++++++++++++++++++++++++++++++++++++++++++++")
        logging.info("+++++++++++   REACHED MAX COMP LIMIT   +++++++++++")
        logging.info("++++++++++++++++++++++++++++++++++++++++++++++++++")
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
                em.fit_many_comps(star_pars, ncomps, rdir=run_dir, pool=pool,
                                  bg_ln_ols=bg_ln_ols) # kill component here
            new_groups = np.array(new_groups)

        new_lnlike = em.get_overall_lnlikelihood(star_pars, new_groups,
                                                 bg_ln_ols=bg_ln_ols)
        new_lnpost = em.get_overall_lnlikelihood(star_pars, new_groups,
                                                 bg_ln_ols=bg_ln_ols,
                                                 inc_posterior=True)
        new_BIC = em.calc_bic(star_pars, ncomps, new_lnlike)
    # handle multiple components
    else:
        logging.info("******************************************")
        logging.info("*********  FITTING {} COMPONENTS  *********".\
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
                    em.fit_many_comps(star_pars, ncomps, rdir=run_dir,
                                      pool=pool,
                                      init_comps=init_groups,
                                      bg_ln_ols=bg_ln_ols)
            best_fits.append(groups)
            all_meds.append(meds)
            all_zs.append(z)

            lnlike = em.get_overall_lnlikelihood(star_pars, groups,
                                                 bg_ln_ols=bg_ln_ols)
            lnlikes.append(lnlike)
            lnposts.append(em.get_overall_lnlikelihood(star_pars, groups,
                                                       bg_ln_ols=bg_ln_ols,
                                                       inc_posterior=True))
            BICs.append(em.calc_bic(star_pars, ncomps, lnlike))
            logging.info("Decomp finished with\nBIC: {}\nLnlike: {}"
                         "\nlnpost: {}".format(
                BICs[-1], lnlikes[-1], lnposts[-1]
            ))

        # identify the best performing decomposition
        # best_split_ix = np.argmax(lnposts)
        best_split_ix = np.argmin(BICs)
        new_groups, new_meds, new_z, new_lnlike, new_lnpost, new_BIC = \
            zip(best_fits, all_meds, all_zs,
                lnlikes, lnposts, BICs)[best_split_ix]
        logging.info("Selected {} as best decomposition".format(best_split_ix))
        logging.info("Turned\n{}".format(
            prev_groups[best_split_ix].getInternalSphericalPars()))
        logging.info("into\n{}\n&\n{}".format(
            new_groups[best_split_ix].getInternalSphericalPars(),
            new_groups[best_split_ix + 1].getInternalSphericalPars(),
        ))

    # Check if the fit has improved
    if new_BIC < prev_BIC:
        logging.info("Extra component has improved BIC...")
        logging.info("New BIC: {} < Old BIC: {}".format(new_BIC, prev_BIC))
        logging.info("lnlike: {} | {}".format(new_lnlike, prev_lnlike))
        logging.info("lnpost: {} | {}".format(new_lnpost, prev_lnpost))
        prev_groups, prev_meds, prev_z, prev_lnlike, prev_lnpost, \
        prev_BIC = \
            (new_groups, new_meds, new_z, new_lnlike, new_lnpost, new_BIC)
        ncomps += 1
    else:
        logging.info("Extra component has worsened BIC...")
        logging.info("New BIC: {} > Old BIC: {}".format(new_BIC, prev_BIC))
        logging.info("lnlike: {} | {}".format(new_lnlike, prev_lnlike))
        logging.info("lnpost: {} | {}".format(new_lnpost, prev_lnpost))
        logging.info("... saving previous fit as best fit to data")
        np.save(rdir+'final_best_groups.npy', prev_groups)
        np.save(rdir+'final_med_errs.npy', prev_meds)
        np.save(rdir+'final_membership.npy', prev_z)
        np.save(rdir+'final_likelihood_post_and_bic', [prev_lnlike, prev_lnpost,
                                                       prev_BIC])
        logging.info('Final best fits:')
        [logging.info(g.getSphericalPars()) for g in prev_groups]
        logging.info('Final age med and span:')
        [logging.info(row[-1]) for row in prev_meds]
        logging.info('Membership distribution: {}'.format(prev_z.sum(axis=0)))
        logging.info('Final membership:')
        logging.info('\n{}'.format(np.round(prev_z*100)))
        logging.info('Final lnlikelihood: {}'.format(prev_lnlike))
        logging.info('Final lnposterior:  {}'.format(prev_lnpost))
        logging.info('Final BIC: {}'.format(prev_BIC))
        break

    logging.info("Best fit:\n{}".format(
        [group.getInternalSphericalPars() for group in prev_groups]))

if using_mpi:
    pool.close()
