#! /usr/bin/env python
from __future__ import print_function, division

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
import chronostar.synthesiser as syn
import chronostar.traceorbit as torb
import chronostar.converter as cv
import chronostar.measurer as ms
import chronostar.expectmax as em
import chronostar.fitplotter as fp
import chronostar.datatool as dt

try:
    run_name = sys.argv[1]
except:
    run_name = 'dummy'

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
mean_now = np.array([50., -100., -0., -10., -20., -5.])
extra_pars = np.array([
    #dX, dV, age, nstars
    [10., 0.5,  2.,  10.],
    [ 2., 1.0,  5.,  15.],
    # [ 5., 0.7, 10.,  50.],
    # [100., 50.,  1e-5, 1000.],
])
logging.info("Mean (now):\n{}".format(mean_now))
logging.info("Extra pars:\n{}".format(extra_pars))
ERROR = 1.0
ngroups = extra_pars.shape[0]

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
        mean_now_w_offset = mean_now.copy()
        mean_now_w_offset[0] += i * 50
    
        mean_then = torb.traceOrbitXYZUVW(mean_now_w_offset, -extra_pars[i,-2],
                                          single_age=True)
        group_pars = np.hstack((mean_then, extra_pars[i]))
        xyzuvw_init, origin = syn.synthesiseXYZUVW(group_pars, sphere=True,
                                                   return_group=True,
                                                   internal=False)
        origins.append(origin)
        all_xyzuvw_init = np.vstack((all_xyzuvw_init, xyzuvw_init))
        xyzuvw_now_perf = torb.traceManyOrbitXYZUVW(xyzuvw_init,
                                                    times=origin.age,
                                                    single_age=True)
        all_xyzuvw_now_perf = np.vstack((all_xyzuvw_now_perf, xyzuvw_now_perf))
    
    np.save(groups_savefile, origins)
    np.save(xyzuvw_perf_file, all_xyzuvw_now_perf)
    astro_table = ms.measureXYZUVW(all_xyzuvw_now_perf, 1.0,
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
            new_groups, new_meds, new_z =\
                em.fitManyGroups(star_pars, ncomps, rdir=run_dir, pool=pool)
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
            logging.info("-------- decomposing {}th component ---------".\
                         format(i))
            run_dir = rdir + '{}/{}/'.format(ncomps, chr(ord('A') + i))
            mkpath(run_dir)

            # Decompose and replace the ith group with two new groups
            # by using the 16th and 84th percentile ages from chain
            _, split_groups = em.decomposeGroup(split_group,
                                                young_age = prev_meds[i,-1,1],
                                                old_age = prev_meds[i,-1,2])
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
                    em.fitManyGroups(star_pars, ncomps, rdir=run_dir, pool=pool,
                                     init_groups=init_groups)
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
            new_groups[best_split_ix+1].getInternalSphericalPars(),
        ))

    # Check if the fit has improved
    if new_lnpost > prev_lnpost:
        logging.info("Extra component has improved lnpost...")
        logging.info("{} > {}".format(new_lnpost, prev_lnpost))
        logging.info("New BIC: {} | Old BIC: {}".format(new_BIC, prev_BIC))
        prev_groups, prev_meds, prev_z, prev_lnlike, prev_lnpost, prev_BIC =\
            (new_groups, new_meds, new_z, new_lnlike, new_lnpost, new_BIC)
        ncomps += 1
    else:
        logging.info("Extra component has worsened lnpost...")
        break

    logging.info("Best fit:\n{}".format(
        [group.getInternalSphericalPars() for group in prev_groups]))

if using_mpi:
    pool.close()
