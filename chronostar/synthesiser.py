"""
synthesiser
Goal: Have a bunch of synthesised traceback scenarios

Tool: Generate initial conditions of a "group",
project forward in time,
make artificial "measurements",
store as a star table,
save "ground truths" in some log somewhere
"""

from __future__ import print_function, division

import numpy as np
import traceback as tb
import pickle
from astropy.table import Table
import pdb
from utils import generate_cov

# Stored as global constant for ease of comparison in testing suite           
GAIA_ERRS = {
    'e_Plx':0.6, #e_Plx [mas]
    'e_RV' :0.5,  #e_RV [km/s]
    'e_pm' :0.42, #e_pm [mas/yr]
    }

def synth_group(group_pars):
    """Synthesise an association of stars in galactic coords at t=0

    Parameters
    ----------
    group_pars
        [14] array with the following values:
        [X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars]

    Returns
    -------
    xyzuvw_now
        The XYZUVW phase space values of stars at current time
    """
    nstars = int(group_pars[-1])
    age = group_pars[-2]

    # build covariance matrix, using "internal" parametrisation
    group_pars_in = np.copy(group_pars[:-1])
    group_pars_in[6:10] = 1./group_pars_in[6:10]

    # Sample stars' initial parameters
    xyzuvw_init = np.random.multivariate_normal(
        mean=group_pars[0:6], cov=cov, size=nstars
        )

    # Project forward in time
    xyzuvw_now = np.zeros((nstars,6))
    for i in range(nstars):
        xyzuvw_now[i] = tb.trace_forward(xyzuvw_init[i], age,
                                        solarmotion=None)

    return xyzuvw_now

def measure_stars(xyzuvw_now):
    """
    Take a bunch of stars' XYZUVW in the current epoch, and convert into
    observational measurements with perfect precision.

    Parameters
    ----------
    xyzuvw_now
        [nstars,6] array with synthesised XYZUVW data

    Returns
    -------
    sky_coord_now
        [nstars,6] array with synthesised measurements:
        {RA, DEC, pi, pmRA, pmDEC, RV}
    """
    # convert to radecpipmrv coordinates:
    nstars = xyzuvw_now.shape[0]
    sky_coord_now = np.zeros((nstars,6))
    for i in range(nstars):
        sky_coord_now[i] = tb.xyzuvw_to_skycoord(
            xyzuvw_now[i], solarmotion='schoenrich', reverse_x_sign=True
            )
    return sky_coord_now

def synthesise_data(ngroups, group_pars, error, savefile=None):
    """
    Entry point of module; synthesise the observational measurements of an
    arbitrary number of groups with arbitrary initial conditions, with 
    arbitrary degree of precision. Saving the data as an astropy table.

    Input
    -----
    ngroups
        Number of groups
    group_pars
        either [15] or [ngroups,15] array of parameters describing
        the initial conditions of a group. NOTE, group_pars[-1] is nstars
        {X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars}
    error
        float [0,1+], degree of precision in our "instruments" linearly 
        ranging from perfect (0) to Gaia-like (1)
    savefile
        optional name for output file

    Output
    ------
    * a saved astropy table: data/synth_[N]groups_[N]stars.pkl
    """

    # For each group, generate current XYZUVW positions
    if ngroups == 1:
        xyzuvw_init = synth_group(group_pars)
        nstars = int(group_pars[-1])

    else:
        xyzuvw_init = np.zeros((0,6))
        for i in range(ngroups):
            xyzuvw_init = np.vstack( (xyzuvw_init, synth_group(group_pars[i])) )
        nstars = int(np.sum(group_pars[:,-1]))

    # compile sky coordinates into a table with some form of error
    sky_coord_now = measure_stars(xyzuvw_init, nstars)

    # based off projected Gaia goal and GALAH(??)
    e_plx = GAIA_ERRS['e_Plx'] #0.6 #mas
    e_RV  = GAIA_ERRS['e_RV'] #0.5 #km/s
    e_pm  = GAIA_ERRS['e_pm'] #0.42 #mas/yr

    errs = np.ones(nstars) * error

    ids = np.arange(nstars)
    # note, x + error*x*N(0,1) == x * N(1,error)
    # i.e., we resample the measurements based on the measurement
    # 'error'
    t = Table(
        [
        ids,                #names
        sky_coord_now[:,0], #RAdeg
        sky_coord_now[:,1], #DEdeg
        np.random.normal(sky_coord_now[:,2], e_plx*error),  #Plx [mas]
        e_plx * errs,
        np.random.normal(sky_coord_now[:,5], e_RV*error),   #RV [km/s]
        e_RV * errs,
        np.random.normal(sky_coord_now[:,3], e_pm*error),   #pmRA [mas/yr]
        e_pm * errs,
        np.random.normal(sky_coord_now[:,4], e_pm*error),   #pmDE [mas/yr]
        e_pm * errs,
        ],
        names=('Name', 'RAdeg','DEdeg','Plx','e_Plx','RV','e_RV',
               'pmRA','e_pmRA','pmDE','e_pmDE')
        )
    #times = np.linspace(0,30,31)

    if savefile is None:
        savefile = "data/synth_data_{}groups_{}stars{}err.pkl".\
                format(ngroups, nstars, int(100*error))
    pickle.dump(t, open(savefile, 'w'))
    #print("Synthetic data file successfully created")

    try:
        # keep track of initial group_pars for each synthetic traceback set
        with open("data/synth_log.txt", 'a') as logfile:
            logfile.write("\n------------------------\n")
            logfile.write(
                "filename: {}\ngroup parameters [X,Y,Z,U,V,W,dX,dY,dZ,dV,"
                "Cxy,Cxz,Cyz,age,nstars]:\n{}\nerror: {}\n".\
                format(savefile, group_pars,error))
    except IOError:
        pass

if __name__ == '__main__':
    """ simple, sample usage """
    group_pars = np.array([
        [0,0,0,0,0,0,10,10,10,3,0.1,0.2,0.0,20,50],
        [0,0,0,0,0,0,30,10,10,3,0.1,0.2,0.0,20,50],
        ])
    synthesise_data(2, group_pars, 0.01)
