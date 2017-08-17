#! /usr/bin/env python

from __future__ import division, print_function
import numpy as np
import chronostar.traceback as tb
from astropy.table import Table
import pickle
import matplotlib.pyplot as plt
import chronostar.error_ellipse as ee
import pdb

init_age = 20
traceback_age = 20
init_centre   = np.array([5, 60, 25, -0.12, -3.48, 1.06])
init_pos_disp = 10
init_vel_disp = 3 
perc_error = 0.001

xycorr = 0.0
xzcorr = 0.0
yzcorr = 0.0

corrs = [xycorr, xzcorr, yzcorr]
stdevs = [init_pos_disp, init_pos_disp, init_pos_disp,
          init_vel_disp, init_vel_disp, init_vel_disp]

cov = np.eye(6)
cov[np.tril_indices(3,-1)] = corrs
cov[np.triu_indices(3,1)] = corrs

for i in range(3):
    cov[:3,i] *= stdevs[:3]
    cov[i,:3] *= stdevs[:3]

for i in range(3,6):
    cov[3:6,i] *= stdevs[3:]
    cov[i,3:6] *= stdevs[3:]

print(cov)

np.random.seed(0)
nstars = 30 

# generate initial stars from an arbitrary covariance matrix
xyzuvw_init = np.random.multivariate_normal(
        mean=init_centre, cov=cov, size=nstars
        )

savefile = "synt_traceback_{}Myr_{}stars.pkl".format(
                traceback_age, nstars)
# Generate the initial xyzuvw data of a group
# the group is isotropic in position and velocity

# Manual calculation left here for sanity reasons
np.random.seed(0)
if(False):
    random = np.random.normal(size=(nstars,6))
    xyzuvw_init = np.zeros((nstars,6))
    xyzuvw_init[:,:3] = random[:,:3]*init_pos_disp
    xyzuvw_init[:,3:] = random[:,3:]*init_vel_disp
    xyzuvw_init = xyzuvw_init + init_centre

# Project forward in time
xyzuvw_now = np.zeros((nstars,6))
for i in range(nstars):
    xyzuvw_now[i] = tb.trace_forward(xyzuvw_init[i], init_age,
                                     solarmotion=None)

# Convert to radecpipmrv coordinates:
sky_coord_now = np.zeros((nstars,6))
for i in range(nstars):
    sky_coord_now[i] = tb.xyzuvw_to_skycoord(
        xyzuvw_now[i], solarmotion='schoenrich', reverse_x_sign=True
        )

# compile sky coordinates into a table with some form of error

ids = np.arange(nstars)
t = Table(
    [
    ids,
    sky_coord_now[:,0],
    sky_coord_now[:,1],
    sky_coord_now[:,2],
    sky_coord_now[:,2] * perc_error,
    sky_coord_now[:,5],
    sky_coord_now[:,5] * perc_error,
    sky_coord_now[:,3],
    sky_coord_now[:,3] * perc_error,
    sky_coord_now[:,4],
    sky_coord_now[:,4] * perc_error
    ],
    names=('Name', 'RAdeg','DEdeg','Plx','e_Plx','RV','e_RV',
           'pmRA','e_pmRA','pmDE','e_pmDE')
    )
times = np.linspace(0,30,31)

# perform traceback
traceback = tb.traceback(t, times, savefile="data/"+savefile)
stars, ages, xyzuvw, xyzuvw_cov = pickle.load(open("data/"+savefile, 'r'))

# plot result
# for each star, plot the traceback curve to its initial age
# as well as a covariance matrix at it's "true" age
plt.clf()
for i in range(nstars):
    x, y = xyzuvw[i,:21,:2].T
    plt.plot(x,y, 'r')

    pltcov = xyzuvw_cov[i,20,:2,:2]

    ee.plot_cov_ellipse(
        cov=pltcov,
        pos=xyzuvw[i,20,:2],
        nstd=10,
        )
plt.xlabel("X [pc]")
plt.ylabel("Y [pc]")
plt.show()
