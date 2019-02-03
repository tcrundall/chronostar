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
    ass_name = 'marusa_galah_li_strong_stars'

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

star_pars = dt.loadXYZUVW(xyzuvw_file)

#print(star_pars)

# Grid
grid_x = [-10000, -40, -10, 10, 40, 10000]
grid_y = [-10000, -10, 10, 10000]
grid_z = [-10000, 0, 10000]

xyzuvw = star_pars['xyzuvw']
print xyzuvw

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
            xyzuvw_box = xyzuvw[mask]

            if len(init_z)<1:
                init_z = mask.astype(int)
            else:
                init_z = np.vstack((init_z, mask.astype(int)))

            print init_z

init_z=init_z.T
print init_z.shape
print np.sum(init_z)