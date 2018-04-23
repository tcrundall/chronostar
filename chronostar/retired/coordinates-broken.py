from __future__ import print_function, division

import logging
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from galpy.util import bovy_coords


def convertLSRToRADEC(xyzuvw_lsr, kpc=False):
    """
    Generate astrometry values from cartesian coordinates in LSR frame

    Paramters
    ---------
    xyzuvw_lsr : [6] (or [nstars, 6] ???) array
        [pc, pc, pc, km/s, km/s, km/s]
    kpc : boolean {False}
        set this flag if input is in kpc not pc

    Returns
    -------

    """
    if not kpc:
        xyzuvw_lsr[:3] /= 1e3
    xyzuvw_hc = convertLSRToHelioCentric(xyzuvw_lsr, kpc=True)
    return convertHelioCentricToRADEC(xyzuvw_hc)

def convertHelioCentricToRADEC(xyzuvw_hc, kpc=False):
    """
    Generate astrometry values from cartesian coordinates centred on sun

    Parameters
    ----------
    xyzuvw_hc : [6] array
        [kpc, kpc, kpc, km/s, km/s, km/s]

    Returns
    -------
    astrometry : [6] array
        [RA, DEC, pi, pm_ra, pm_dec, vr]
    """
#    if not kpc:
#        xyzuvw_hc = xyzuvw_hc.copy()
#        xyzuvw_hc[:3] /= 1e3
    logging.debug("Positions is: {}".format(xyzuvw_hc[:3]))
    logging.debug("Velocity is: {}".format(xyzuvw_hc[3:]))
    lbdist = convertHelioCentricTolbdist(xyzuvw_hc)
    radec = bovy_coords.lb_to_radec(lbdist[0], lbdist[1], degree=True)
    vrpmllpmbb = bovy_coords.vxvyvz_to_vrpmllpmbb(
        xyzuvw_hc[3], xyzuvw_hc[4], xyzuvw_hc[5],
        lbdist[0], lbdist[1], lbdist[2],
        degree=True
    )
    pmrapmdec = bovy_coords.pmllpmbb_to_pmrapmdec(
        vrpmllpmbb[1], vrpmllpmbb[2], lbdist[0], lbdist[1], degree=True
    )
    return [radec[0], radec[1], 1.0 / lbdist[2],
            pmrapmdec[0], pmrapmdec[1], vrpmllpmbb[0]]


def convertLSRToHelioCentric(xyzuvw_lsr):
    """
    Converting XYZUVW values from LSR centred coords to heliocentric coords

    Galpy coordinates are from the vantage point of the sun.
    So even though the sun changes position with time,  we still "observe" the
    star from the current solar position and motion regardless of the traceback
    or traceforward time

    Parameters
    ----------
    xyzuvw_lsr : [6] (or [npoints, 6]) float array
        The XYZUVW values (at various points) with respect to the local
        standard of rest.
        NOTE UNITS: [kpc, kpc, kpc, km/s, km/s, km/s]

    Returns
    -------
    xyzuvw_helio : [6] (or [npoints, 6]) float array
        The XYZUVW values (at various points) with respect to the solar position
        and motion
    """
    XYZUVWSOLARNOW = np.array([0., 0., 0.025, 11.1, 12.24, 7.25])
    return xyzuvw_lsr - XYZUVWSOLARNOW

def convertHelioCentricTolbdist(xyz):
    """Convert position from heliocentric coordinates into galactic coordinates

    The unit of distance will match the units put in. If xyz is 2 dimensional,
    the output will also be 2 dimensional

    Parameters
    ----------
    xyz : [3] (or [npoints, 3]) float array
        The position of the star in standard chronostar coordinates. Can
        be two dimensional (positions of many stars) if desired.

    Returns
    -------
    l : float
        in degrees, galactic longitude angular position through galactic
        plane (0 = GC, 90 = direction of circular velocity)
    b : float
        in degrees, galactic latitude, angle of object above galactic plane
        as measured from the Sun
    dist : float
        in units of input, distance from Sun to object
    OR
    res : [nstars, 3] float array
    """
    xyz = np.array(xyz)
    multiple_stars = True
    if len(xyz.shape) == 1:
        xyz = xyz.reshape((1, -1))
        multiple_stars = False
    l = np.degrees(np.arctan2(xyz[:,1], xyz[:,0]))
    b = np.degrees(np.arctan2(xyz[:,2], np.sqrt(np.sum(xyz[:,:2]**2, axis=1))))
    dist = np.sqrt(np.sum(xyz[:,:3]**2, axis=1))
    if multiple_stars:
        res = np.vstack((l,b,dist)).T
    else:
        res = (l[0], b[0], dist[0])
    return res