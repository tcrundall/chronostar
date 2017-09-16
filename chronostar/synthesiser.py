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
import pickle
#import chronostar.traceback as tb
import traceback as tb #??? why won't this line work?
from astropy.table import Table
import pdb

def synth_group(params):
    """
    Input
    -----
    params: [X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars]
    perc_error: the percentage error of mock observational measurements
    """
    nstars = int(params[-1])
    age = params[-2]
    
    # build covariance matrix
    cov = np.eye(6)
    cov[np.tril_indices(3,-1)] = params[10:13]
    cov[np.triu_indices(3, 1)] = params[10:13]
    
    for i in range(3):
        cov[:3,i] *= params[6:9]
        cov[i,:3] *= params[6:9]

    for i in range(3,6):
        cov[3:6,i] *= params[9]
        cov[i,3:6] *= params[9]

    # Sample stars' initial parameters
    xyzuvw_init = np.random.multivariate_normal(
        mean=params[0:6], cov=cov, size=nstars
        )

    # Project forward in time
    xyzuvw_now = np.zeros((nstars,6))
    for i in range(nstars):
        xyzuvw_now[i] = tb.trace_forward(xyzuvw_init[i], age,
                                        solarmotion=None)

    return xyzuvw_now

def measure_stars(xyzuvw_now, nstars):
    """
    !!!TO DO:
        Deviate the "measurements" by the prescribed error

    Take a bunch of stars' XYZUVW in the current epoch, and convert into
    observational measurements with some uncertainty.

    Input
    -----
    xyzuvw_now: a [nstars,6] array with synthesised XYZUVW data
    nstars: number of stars

    Output
    ------
    sky_coord_now: [nstars,6] array with synthesised measurements:
        {RA, DEC, pi, pmRA, pmDEC, RV}
    """
    # convert to radecpipmrv coordinates:
    sky_coord_now = np.zeros((nstars,6))
    for i in range(nstars):
        sky_coord_now[i] = tb.xyzuvw_to_skycoord(
            xyzuvw_now[i], solarmotion='schoenrich', reverse_x_sign=True
            )
    return sky_coord_now

def synthesise_data(ngroups, group_pars, error):
    """
    Entry point of module; synthesise the observational measurements of an
    arbitrary number of groups with arbitrary initial conditions, with 
    arbitrary degree of precision. Saving the data as an astropy table.

    Input
    -----
    ngroups: Number of groups
    group_pars: either [15] or [ngroups,15] array of parameters describing
        the initial conditions of a group. NOTE, group_pars[-1] is nstars
        {X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars}
    error: float [0,1+], degree of precision in our "instruments" linearly 
        ranging from perfect (0) to Gaia-like (1)

    Output
    ------
    * a saved astropy table: data/synth_[N]groups_[N]stars.pkl

    todo:
        have uncertainites on the order of Gaia
        PM 20 micro arcsec / yr
        RV 1 km/s
        par 20 micro arcsec

        WHAT UNITS ARE BOVY COORDS GIVEN IN?
        I think milli arcsec...
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
    e_plx = 0.7 #mas
    e_pm  = 3.2 #mas/yr
    e_RV  = 1.3 #km/s

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
        np.random.normal(sky_coord_now[:,2], e_plx*error),  #Plx
        e_plx * errs,
        np.random.normal(sky_coord_now[:,5], e_RV*error),   #RV
        e_RV * errs,
        np.random.normal(sky_coord_now[:,3], e_pm*error),   #pmRA
        e_pm * errs,
        np.random.normal(sky_coord_now[:,4], e_pm*error),   #pmDE
        e_pm * errs,
        ],
        names=('Name', 'RAdeg','DEdeg','Plx','e_Plx','RV','e_RV',
               'pmRA','e_pmRA','pmDE','e_pmDE')
        )
    #times = np.linspace(0,30,31)

    savefile = "synth_data_{}groups_{}stars.pkl".format(ngroups, nstars)
    pickle.dump(t, open("data/" + savefile, 'w'))
    print("Synthetic data file successfully created")

    with open("data/synth_log.txt", 'a') as logfile:
        logfile.write("\n------------------------\n")
        logfile.write(
            "filename: {}\ngroup parameters [X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,"
            "Cyz,age,nstars]:\n{}\nerror: {}\n".\
            format(savefile, group_pars,error))

if __name__ == '__main__':
    """ simple, sample usage """
    group_pars = np.array([
        [0,0,0,0,0,0,10,10,10,3,0.1,0.2,0.0,20,50],
        [0,0,0,0,0,0,30,10,10,3,0.1,0.2,0.0,20,50],
        ])
    synthesise_data(2, group_pars, 0.01)
