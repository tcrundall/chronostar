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
import chronostar.synthdata as syn
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



star_pars = dt.loadXYZUVW(xyzuvw_file)

np.set_printoptions(suppress=True)

# --------------------------------------------------------------------------
# Get grid-based Z membership
# --------------------------------------------------------------------------

# Grid
xyzuvw = star_pars['xyzuvw']
dmin=np.min(xyzuvw, axis=0)-1
dmax=np.max(xyzuvw, axis=0)+1
grid_x = np.linspace(dmin[0], dmax[0], 10) #[-10000, -60, -20, 20, 60, 10000]
grid_y = np.linspace(dmin[1], dmax[1], 10)
grid_z = np.linspace(dmin[2], dmax[2], 6)


ncomps=(len(grid_x)-1)*(len(grid_y)-1)*(len(grid_z)-1)
print('ncomps: %d'%ncomps)

# Create init_z
init_z=[]
for x1, x2 in zip(grid_x[:-1], grid_x[1:]):
    for y1, y2 in zip(grid_y[:-1], grid_y[1:]):
        for z1, z2 in zip(grid_z[:-1], grid_z[1:]):
            mask = (xyzuvw[:,0]>x1) & (xyzuvw[:,0]<=x2)
            mask = mask & (xyzuvw[:,1]>y1) & (xyzuvw[:,1]<=y2)
            mask = mask & (xyzuvw[:,2]>z1) & (xyzuvw[:,2]<=z2)

            if len(init_z)<1:
                init_z = mask.astype(int)
            else:
                init_z = np.vstack((init_z, mask.astype(int)))
# Add a column for the background

init_z=init_z.T

print(np.sum(init_z, axis=0))

print('init_z successful!! Yey')

# Delete components with no stars
nc=np.sum(init_z, axis=0)
mask=nc>1
print(nc)
print(nc[mask])
#print(init_z[mask])
#print(len(init_z[mask]))
print(nc.shape)
init_z=init_z[:,mask]
init_z=init_z.T
init_z = np.vstack((init_z, np.zeros(len(xyzuvw))))
init_z=init_z.T
print(init_z)