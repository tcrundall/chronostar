from __future__ import division, print_function
import numpy as np
import chronostar.traceback as tb
from astropy.table import Table

init_age = 20
init_centre   = np.array([5, 60, 25, -0.12, -3.48, 1.06])
init_pos_disp = 10
init_vel_disp = 2
nstars = 5 
np.random.seed(0)
random = np.random.normal(size=(nstars,6))

# Generate the initial xyzuvw data of a group
# the group is isotropic in position and velocity
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

perc_error = 0.001

errors = np.zeros(nstars) + perc_error

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

# trace back
traceback = tb.traceback(t, times, savefile="data/synthetic_tb.pkl")


