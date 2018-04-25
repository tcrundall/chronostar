"""
traceorbit.py

A module aimed at projecting an orbit forward or backward through time.
Operates in a co-rotating, RH cartesian coordinate system centred on the local
standard of rest.
"""

import logging
import numpy as np

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014 as mp
from galpy.util import bovy_conversion

import coordinate as cc


def convertToBovyTime(times):
    """
    Convert times provided in Myr into times in bovy internal units

    Paramters
    ---------
    times : [ntimes] float array
        Times in Myr

    Return
    ------
    bovy_times : [ntimes] float array
        Times in bovy internal units
    """
    bovy_times = times*1e-3 / bovy_conversion.time_in_Gyr(220., 8.)
    return bovy_times


def convertGalpyCoordsToXYZUVW(data, ts=None, ro=8., vo=220., rc=True):
    """
    Converts orbits from galpy internal coords to chronostar coords

    Data should be raw galpy data (i.e. output from o.getOrbit()).
    Chronostar coordinate frame is a corotating reference frame centred on
    the LSR as defined by the Schoenrich solar motion of
    XYZUVW = 0, 0, 25pc, 11.1 km/s, 12.24 km/s, 7.25 km/s
    Galpy coordinates are [R, vR, vT, z, vz, phi]
    By default, positions are scaled by LSR distance from galactic centre,
    ro=8kpc, and velocities scaled by the LSR circular velocity, vo = 220km/s.
    Time is scaled such that after 1 time unit has passed, the LSR has travelled
    1 radian about the galactic centre. The values are returned in a [ntimes, 6]
    array:
        R : galactic radial distance /ro
        vR : galactic radial velocity /vo
        vT : circular velocity /vo
        z  : vertical distance from plane / ro
        vz : vertical velocity / vo
        phi : angle about the galaxy (anticlockwise from LSR's location at t=0)
    For example, the LSR at t=0.0 and t=1.0 as values:
    [1., 0., 1., 0., 0., 0.]
    [1., 0., 1., 0., 0., 1.]

    Parameters
    ----------
    data : [ntimes, 6] float array
        output from o.getOrbit. Data is encoded as:
        [R, vR, vT, z, vz, phi]
        R : galactic radial distance /ro
        vR : galactic radial velocity /vo
        vT : circular velocity /vo
        z  : vertical distance from plane / ro
        vz : vertical velocity / vo
        phi : angle about the galaxy (anticlockwise from LSR's location at t=0)
    ts : [ntimes] float array [galpy time units]
        times used to generate orbit. Ensure the units are in galpy time units
    ro : float
        a conversion factor that takes units from galpy units to
        physical units. If left as default, output will be in kpc
    vo : float
        a conversion factor that takes units form galpy units to
        physical units. If left as default, output will be in km/s
        This is also the circular velocity of a circular orbit with X,Y equal to that of the sun.
    rc : boolean
        whether to calculate XYZUVW in a right handed coordinate system
        (X, U positive towards galactic centre)

    Returns
    -------
    xyzuvw : [ntimes, 6] float array
        [pc, pc, pc, km/s, km/s, km/s] - traced orbit in chronostar coordinates
    """
    if ts is not None:
        phi_lsr = ts
    else:
        phi_lsr = 0.0
    R, vR, vT, z, vz, phi_s = data.T

    # This is the angular distance between the LSR and our star
    phi = phi_s - phi_lsr

    # Can convert to XYZUVW coordinate frame. See thesis for derivation
    # Need to scale values back into physical units with ro and vo.
    # 1. in X and V are the LSR R and vT respectively (which are unitary due to
    # the normalisation of units inside galpy
    X = 1000 * ro * (1. - R * np.cos(phi))
    Y = 1000 * ro * R * np.sin(phi)
    Z = 1000 * ro * z
    U = vo * (-vR*np.cos(phi) + vT*np.sin(phi))
    V = vo * ( vT*np.cos(phi) + vR*np.sin(phi) - 1.)
    W = vo * vz

    if not rc:
        print("BUT EVERYONE IS USING RHC!!!")
        X = -X
        U = -U

    xyzuvw = np.vstack((X,Y,Z,U,V,W)).T
    # included for compatability with single data point
    if xyzuvw.shape == (1,6):
        xyzuvw = xyzuvw[0]
    return xyzuvw

def traceOrbitXYZUVW(xyzuvw_start, times):
    """
    Given a star's XYZUVW relative to the LSR (at any time), project its
    orbit forward (or backward) to each of the times listed in *times*

    Positive times --> traceforward
    Negative times --> traceback

    Parameters
    ----------
    xyzuvw : [pc,pc,pc,km/s,km/s,km/s]
    times : [ntimes] float array
        Myr - time of 0.0 must be present in the array. Times need not be
        spread linearly.

    Returns
    -------
    xyzuvw_tf : [ntimes, 6] array
        [pc, pc, pc, km/s, km/s, km/s] - the traced orbit with positions
        and velocities

    TODO: allow for the input of a single age, instead of time array
    """
    # convert positions to kpc
    xyzuvw_start = np.copy(xyzuvw_start)
    xyzuvw_start[:3] *= 1e-3
    bovy_times = convertToBovyTime(times)
    logging.debug("Tracing up to {} Myr".format(times[-1]))
    logging.debug("Tracing up to {} Bovy yrs".format(bovy_times[-1]))

    logging.debug("Initial lsr start: {}".format(xyzuvw_start))
    xyzuvw_helio = cc.convertLSRToHelio(xyzuvw_start, kpc=True)
    logging.debug("Initial helio start: {}".format(xyzuvw_helio))
    logging.debug("Galpy vector: {}".format(xyzuvw_helio))

    l,b,dist = cc.convertCartesianToAngles(
        *xyzuvw_helio[:3], return_dist=True, value=True
    )
    #l,b,dist = cc.convertHelioCentricTolbdist(xyzuvw_helio)
    vxvv = [l,b,dist,xyzuvw_helio[3],xyzuvw_helio[4],xyzuvw_helio[5]]
    logging.debug("vxvv: {}".format(vxvv))
    o = Orbit(vxvv=vxvv, lb=True, uvw=True, solarmotion='schoenrich')

    o.integrate(bovy_times,mp,method='odeint')
    data_gp = o.getOrbit()
    xyzuvw = convertGalpyCoordsToXYZUVW(data_gp, bovy_times)
    logging.debug("Started orbit at {}".format(xyzuvw[0]))
    logging.debug("Finished orbit at {}".format(xyzuvw[-1]))
    return xyzuvw

def traceManyOrbitXYZUVW(xyzuvw_starts, times):
    """
    Given a star's XYZUVW relative to the LSR (at any time), project its
    orbit forward (or backward) to each of the times listed in *times*

    Positive times --> traceforward
    Negative times --> traceback

    Parameters
    ----------
    xyzuvw_starts : [nstars, 6] array (pc,pc,pc,km/s,km/s,km/s)
    times : [ntimes] float array
        Myr - time of 0.0 must be present in the array. Times need not be
        spread linearly.

    Returns
    -------
    xyzuvw_to : [nstars, ntimes, 6] array
        [pc, pc, pc, km/s, km/s, km/s] - the traced orbit with positions
        and velocities
    """
    times = np.array(times)
    ntimes = times.shape[0]

    nstars = xyzuvw_starts.shape[0]

    xyzuvw_to = np.zeros((nstars, ntimes, 6))

    for st_ix in range(nstars):
        xyzuvw_to[st_ix] = traceOrbitXYZUVW(xyzuvw_starts[st_ix], times)

    return xyzuvw_to
