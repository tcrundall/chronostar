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
        [15] array with the following values:
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
    cov = generate_cov(group_pars_in)

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

def generate_current_pos(ngroups, group_pars):
    """Generate a set of stars at group position and project to modern era

    Parameters
    ----------
    ngroups
        Number of groups being synthesised

    group_pars
        The parametrisation of the group's initial condition
    """
    # For each group, generate current XYZUVW positions
    if ngroups == 1:
        xyzuvw_now = synth_group(group_pars)
        nstars = int(group_pars[-1])

    else:
        xyzuvw_now = np.zeros((0,6))
        for i in range(ngroups):
            xyzuvw_now = np.vstack( (xyzuvw_now, synth_group(group_pars[i])) )
        nstars = int(np.sum(group_pars[:,-1]))
    return xyzuvw_now, nstars

def generate_table_with_error(sky_coord_now, error_perc):
    """Generate an "astrometry" table based on current coords and error

    Parameters
    ----------
    sky_coord_now - [nstars, 6] array
        Sky coordinates (RA, Dec, pi, pmRA, pmDE, RV) of all synthetic stars

    error_perc - float
        Percentage of gaia DR2-esque error. 1e-5 --> barely any measuremnt
        error. 1.0 --> gaia DR2 typical error

    Output
    ------
    t - table of synthetic astrometry table
    """
    # based off projected Gaia goal and GALAH(??)
    e_plx = GAIA_ERRS['e_Plx'] #0.6 #mas
    e_RV  = GAIA_ERRS['e_RV'] #0.5 #km/s
    e_pm  = GAIA_ERRS['e_pm'] #0.42 #mas/yr

    nstars = sky_coord_now.shape[0]

    errs = np.ones(nstars) * error_perc
    ids = np.arange(nstars)
    # note, x + error_perc*x*N(0,1) == x * N(1,error_perc)
    # i.e., we resample the measurements based on the measurement
    # 'error_perc'
    t = Table(
        [
        ids,                #names
        sky_coord_now[:,0], #RAdeg
        sky_coord_now[:,1], #DEdeg
        np.random.normal(sky_coord_now[:,2], e_plx*error_perc),  #Plx [mas]
        e_plx * errs,
        np.random.normal(sky_coord_now[:,5], e_RV*error_perc),   #RV [km/s]
        e_RV * errs,
        np.random.normal(sky_coord_now[:,3], e_pm*error_perc),   #pmRA [mas/yr]
        e_pm * errs,
        np.random.normal(sky_coord_now[:,4], e_pm*error_perc),   #pmDE [mas/yr]
        e_pm * errs,
        ],
        names=('Name', 'RAdeg','DEdeg','Plx','e_Plx','RV','e_RV',
               'pmRA','e_pmRA','pmDE','e_pmDE')
        )
    return t

def synthesise_data(ngroups, group_pars, error_perc, savefile=None):
    """
    Entry point of module; synthesise the observational measurements of an
    arbitrary number of groups with arbitrary initial conditions, with 
    arbitrary degree of precision. Saving the data as an astropy table.

    Input
    -----
    ngroups : int
        Number of groups
    group_pars : [15] array OR [ngroups, 15] array
        array of parameters describing the initial conditions of a group.
        NOTE, group_pars[-1] is nstars
        {X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars}
    error_perc
        float [0,1+], degree of precision in our "instruments" linearly 
        ranging from perfect (0) to Gaia-like (1)
    savefile
        optional name for output file

    Output
    ------
    * a saved astropy table: data/synth_[N]groups_[N]stars.pkl
    """
    xyzuvw_now, nstars = generate_current_pos(ngroups, group_pars)
    sky_coord_now = measure_stars(xyzuvw_now)
    synth_table = generate_table_with_error(sky_coord_now, error_perc)

    if savefile is None:
        savefile = "data/synth_data_{}groups_{}stars{}err.pkl".\
                format(ngroups, nstars, int(100*error_perc))
    pickle.dump(synth_table, open(savefile, 'w'))
    #print("Synthetic data file successfully created")

    try:
        # keep track of initial group_pars for each synthetic traceback set
        with open("data/synth_log.txt", 'a') as logfile:
            logfile.write("\n------------------------\n")
            logfile.write(
                "filename: {}\ngroup parameters [X,Y,Z,U,V,W,dX,dY,dZ,dV,"
                "Cxy,Cxz,Cyz,age,nstars]:\n{}\nerror_perc: {}\n".\
                format(savefile, group_pars,error_perc))
    except IOError:
        pass

if __name__ == '__main__':
    """ simple, sample usage """
    group_pars = np.array([
        [0,0,0,0,0,0,10,10,10,3,0.1,0.2,0.0,20,50],
        [0,0,0,0,0,0,30,10,10,3,0.1,0.2,0.0,20,50],
        ])
    synthesise_data(2, group_pars, 0.01)
