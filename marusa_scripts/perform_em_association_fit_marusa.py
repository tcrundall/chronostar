#! /usr/bin/env python

try:
    import matplotlib as mpl
    mpl.use('Agg')
except ImportError:
    pass

from distutils.dir_util import mkpath
from distutils.errors import DistutilsFileError
import logging
import numpy as np
import pdb
import sys
from emcee.utils import MPIPool
sys.path.insert(0, '..')
import chronostar.expectmax as em
import chronostar.groupfitter as gf
import chronostar.synthesiser as syn
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
try:
    NGROUPS = int(sys.argv[2])
except (IndexError, ValueError):
    NGROUPS = 1

try:
    rdir = "/data/mash/tcrun/em_fit/{}_{}/".format(ass_name.strip('/'),
                                                   NGROUPS)
    gdir = "/data/mash/tcrun/" # directory with master gaia data
    path_msg = "Storing data on mash data server"
    mkpath(rdir)
except (IOError, DistutilsFileError):
    path_msg = ("I'm guessing you're not Tim Crundall..."
                "or not on an RSAA server")
    rdir = "../results/em_fit/{}_{}/".format(ass_name.strip('/'),
                                             NGROUPS)
    gdir = "../data/" # directory with master gaia data
    path_msg = "Storing data on mash data server"
    # if rdir[-1] != '/':
    #     rdir += '/'
    mkpath(rdir)

gaia_xyzuvw_file = gdir + 'gaia_dr2_mean_xyzuvw.npy'
xyzuvw_file = '../data/' + ass_name + '_xyzuvw.fits'
best_fit_file = rdir + "best_fit.npy"
bg_hist_file = "bg_hists.npy"
bg_ln_ols_file = rdir + 'bg_ln_ols.npy'

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

# --------------------------------------------------------------------------
# Calculate background overlap
# --------------------------------------------------------------------------
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
    bg_ln_ols = dt.getKernelDensities(gaia_xyzuvw_file, star_pars['xyzuvw'], amp_scale=0.1)
    np.save(bg_ln_ols_file, bg_ln_ols)
print("continuing")
logging.info("DONE!")

# --------------------------------------------------------------------------
# Get grid-based Z membership
# --------------------------------------------------------------------------

# Grid based on the velocity space
xyzuvw = star_pars['xyzuvw']
dmin=np.min(xyzuvw, axis=0)-1
dmax=np.max(xyzuvw, axis=0)+1
grid_u = np.linspace(dmin[3], dmax[3], 10) #[-10000, -60, -20, 20, 60, 10000]
grid_v = np.linspace(dmin[4], dmax[4], 10)
grid_w = np.linspace(dmin[5], dmax[5], 6)


ncomps=(len(grid_u)-1)*(len(grid_v)-1)*(len(grid_w)-1)
print('ncomps: %d'%ncomps)

# Create init_z
init_z=[]
for u1, u2 in zip(grid_u[:-1], grid_u[1:]):
    for v1, v2 in zip(grid_v[:-1], grid_v[1:]):
        for w1, w2 in zip(grid_w[:-1], grid_w[1:]):
            mask = (xyzuvw[:,3]>u1) & (xyzuvw[:,3]<=u2)
            mask = mask & (xyzuvw[:,4]>v1) & (xyzuvw[:,4]<=v2)
            mask = mask & (xyzuvw[:,5]>w1) & (xyzuvw[:,5]<=w2)

            if len(init_z)<1:
                init_z = mask.astype(int)
            else:
                init_z = np.vstack((init_z, mask.astype(int)))
init_z=init_z.T

# Delete components with no stars
nc=np.sum(init_z, axis=0)
mask=nc>1
init_z=init_z[:,mask]

# Add a column for the background
init_z=init_z.T
init_z = np.vstack((init_z, np.zeros(len(xyzuvw))))
init_z=init_z.T

print(np.sum(init_z, axis=0))

ncomps=len(np.sum(init_z, axis=0))-1
print('ncomps: %d'%ncomps)

print('%d components'%(len(np.sum(init_z, axis=0))-1))
print('init_z successful!! Yey')
# --------------------------------------------------------------------------
# Perform one EM fit
# --------------------------------------------------------------------------
logging.info("Using data file {}".format(xyzuvw_file))
logging.info("Everything loaded, about to fit with {} components"\
    .format(ncomps))
final_groups, final_med_errs, z = em.fitManyGroups(star_pars, ncomps, bg_ln_ols=bg_ln_ols,
                 rdir=rdir, pool=pool, init_z=init_z, ignore_dead_comps=True)


# --------------------------------------------------------------------------
# Repeat EM fit but killing components with too few members
# --------------------------------------------------------------------------
# e.g. maybe < 3 expected star count
# take results from entire past EM fit


#
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
#
# #go through and compare overlap with groups with
# #background overlap
# #
# #bp_pars = np.array([
# #   2.98170398e+01,  4.43573995e+01,  2.29251498e+01, -9.65731744e-01,
# #   -3.42827894e+00, -3.99928052e-02 , 2.63084094e+00,  1.05302890e-01,
# #   1.59367119e+01, nstars
# #])
# #bp_group = syn.Group(bp_pars)
# #
# # --------------------------------------------------------------------------
# # Run fit
# # --------------------------------------------------------------------------
# logging.info("Using data file {}".format(xyzuvw_file))
# logging.info("Everything loaded, about to fit with {} components"\
#     .format(NGROUPS))
# em.fitManyGroups(star_pars, NGROUPS,
#                  rdir=rdir, pool=pool, offset=True, bg_hist_file=bg_hist_file,
#                  origins=origins, init_with_origin=init_origin, init_z=init_z
#                  )
if using_mpi:
    pool.close()
