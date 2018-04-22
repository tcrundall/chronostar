"""
Testing the coordinates.py moduel - a utility module for converting
between various coordinate systems
"""
import numpy as np
import sys

import astropy.units as u
from astropy.coordinates import SkyCoord

from galpy.util import bovy_coords
import logging

sys.path.insert(0,'..')

import chronostar.coordinates as ccoord

def testConvertHelioCentricTolbdist():
    xyz_hc = np.array([
        [0., 0., 0.],
        [10., 0., 0.],
        [0., 10., 0.],
        [0., 0., 10.],
        [10., 10., 0.],
        [10., 10., np.sqrt(200.)],
    ])

    lbdist = np.array([
        [0., 0., 0.],
        [0., 0., 10.],
        [90., 0., 10.],
        [0., 90., 10.],
        [45., 0., np.sqrt(200.)],
        [45., 45., 20.],
    ])

    for xyz, lbd in zip(xyz_hc, lbdist):
        l, b, dist = ccoord.convertHelioCentricTolbdist(xyz)
        assert np.allclose(lbd, np.array((l, b, dist)).reshape(-1))

    assert np.allclose(lbdist, ccoord.convertHelioCentricTolbdist(xyz_hc))


def xyzuvw_to_skycoord(xyzuvw_in, solarmotion='schoenrich',
                       reverse_x_sign=True):
    """Converts XYZUVW with respect to the LSR or the sun
    to RAdeg, DEdeg, plx, pmra, pmdec, RV

    Parameters
    ----------
    xyzuvw_in:
        XYZUVW with respect to the LSR.

    solarmotion: string
        The reference of assumed solar motion. "schoenrich" or None if inputs are already
        relative to the sun.

    reverse_x_sign: bool
        Do we reverse the sign of the X co-ordinate? This is needed for dealing with
        galpy sign conventions.

    TODO: not working at all... fix this
    """
    if solarmotion == None:
        xyzuvw_sun = np.zeros(6)
    elif solarmotion == 'schoenrich':
        xyzuvw_sun = [0, 0, 25, 11.1, 12.24, 7.25]
    else:
        raise UserWarning

    # Make coordinates relative to sun
    xyzuvw = xyzuvw_in - xyzuvw_sun
    logging.debug("Relative to sun: {}".format(xyzuvw))

    # Special for the sun itself...
    # FIXME: This seems like a hack.
    # if (np.sum(xyzuvw**2) < 1) and solarmotion != None:
    #    return [0,0,1e5, 0,0,0]

    # Find l, b and distance.
    # !!! WARNING: the X-coordinate may have to be reversed here, just like
    # everywhere else,
    # because of the convention in Orbit.x(), which doesn't seem to match X.
    if reverse_x_sign:
        lbd = bovy_coords.XYZ_to_lbd(-xyzuvw[0] / 1e3, xyzuvw[1] / 1e3,
                                     xyzuvw[2] / 1e3, degree=True)
    else:
        lbd = bovy_coords.XYZ_to_lbd(xyzuvw[0] / 1e3, xyzuvw[1] / 1e3,
                                     xyzuvw[2] / 1e3, degree=True)
    logging.debug("lbd: {}".format(lbd))
    radec = bovy_coords.lb_to_radec(lbd[0], lbd[1], degree=True)
    logging.debug("radec: {}".format(radec))
    vrpmllpmbb = bovy_coords.vxvyvz_to_vrpmllpmbb(xyzuvw[3], xyzuvw[4],
                                                  xyzuvw[5],
                                                  lbd[0], lbd[1], lbd[2],
                                                  degree=True)
    logging.debug("vr, pm_l, pm_b: {} (km/s, mas/yr, mas/yr)".format(vrpmllpmbb))
    vrpmllpmbb2 = bovy_coords.vxvyvz_to_vrpmllpmbb(xyzuvw[3], xyzuvw[4],
                                                  xyzuvw[5],
                                                  xyzuvw[0]/1e3, xyzuvw[1]/1e3, xyzuvw[2]/1e3,
                                                  XYZ=True)
    logging.debug("vr, pm_l, pm_b: {} (km/s, mas/yr, mas/yr)".format(vrpmllpmbb2))
    pmrapmdec = bovy_coords.pmllpmbb_to_pmrapmdec(vrpmllpmbb[1], vrpmllpmbb[2],
                                                  lbd[0], lbd[1], degree=True)
    logging.debug("pmra, pmdec: {}".format(pmrapmdec))
    return [radec[0], radec[1], 1.0 / lbd[2], pmrapmdec[0], pmrapmdec[1],
            vrpmllpmbb[0]]

#def testConvertLSRToRADEC():
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    bp_ra = '05h47m17.1s'
    bp_dec = '-51d03m59s'
    c = SkyCoord(bp_ra, bp_dec, unit=(u.hourangle, u.deg))
    ra_deg = c.ra.value #deg
    dec_deg = c.dec.value #deg
    mu_ra = 4.65 #mas/yr
    mu_dec = 83.10 #mas/yr
    pi = 51.44 #mas
    dist = 1. / pi #kpc
    vlos = 20. #km/s


    astr_bp = [ra_deg, dec_deg, pi, mu_ra, mu_dec, vlos]
    # [86.82, -51.066, 51.44, 4.65, 83.1, 20.0]


    # should be heliocentric
    xyzuvw_bp = np.array([-3.4, -16.4, -9.9, -11.0, -16.0, 9.1])
    bp_conv = xyzuvw_to_skycoord(xyzuvw_bp, solarmotion=None, reverse_x_sign=False)
    print(bp_conv)
    #ccoord.convertHelioCentricToRADEC(xyzuvw_bp, kpc=False)
