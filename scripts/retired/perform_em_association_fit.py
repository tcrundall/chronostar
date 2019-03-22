#! /usr/bin/env python
import chronostar.component

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
import chronostar.compfitter as gf
import chronostar.synthdata as syn


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

star_pars = gf.loadXYZUVW(xyzuvw_file)

# -------------------------------------------------------------
# constract histograms of Gaia stars in vicinity of association
# -------------------------------------------------------------
logging.info("Building histograms og Gaia stars in vicinity of associaiton")
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
init_origin = False
origins = None
if NGROUPS == 1:
    init_origin = True
    nstars = star_means.shape[0]
    bp_mean = np.mean(star_means, axis=0)
    bp_cov = np.cov(star_means.T)
    bp_dx = np.sqrt(np.min([bp_cov[0,0], bp_cov[1,1], bp_cov[2,2]]))
    bp_dv = np.sqrt(np.min([bp_cov[3,3], bp_cov[4,4], bp_cov[5,5]]))
    bp_age = 0.5
    bp_pars = np.hstack((bp_mean, bp_dx, bp_dv, bp_age, nstars))
    bp_group = chronostar.component.Component(bp_pars)
    origins = [bp_group]

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
logging.info("Using data file {}".format(xyzuvw_file))
logging.info("Everything loaded, about to fit with {} components"\
    .format(NGROUPS))
em.fit_many_comps(star_pars, NGROUPS,
                  rdir=rdir, pool=pool, offset=True, bg_hist_file=bg_hist_file,
                  origins=origins, init_with_origin=init_origin,
                  )
if using_mpi:
    pool.close()
