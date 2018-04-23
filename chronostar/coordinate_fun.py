import logging
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

import sys

a_o = 192.8595 * u.degree
b_ncp = d_o = 27.1283 * u.degree
l_ncp = l_o = 122.9319 * u.degree

old_a_ngp = 192.25 * u.degree
old_d_ngp = 27.4 * u.degree
old_th = 123 * u.degree

def get_l(a, d):
    numer = np.cos(d) * np.sin(a - a_o)
    denom = np.sin(d)*np.cos(d_o) - np.cos(d)*np.sin(d_o)*np.cos(a-a_o)
    return l_o - np.arctan(numer/denom)

a_ngp = 192.25 * u.degree
d_ngp = 27.4 * u.degree

eq_to_gc = np.array([
    [-0.06699, -0.87276, -0.48354],
    [ 0.49273, -0.45035,  0.74458],
    [-0.86760, -0.18837,  0.46020],
])

modern_eq_to_gc = np.array([
    [-0.05487549, -0.87343736, -0.48383454],
    [ 0.49411024, -0.44482901,  0.74698208],
    [-0.86766569, -0.19807659,  0.45598455]
])

modern_gc_to_eq = np.linalg.inv(modern_eq_to_gc)

gc_to_eq = np.linalg.inv(eq_to_gc)



def calcEQToGCMatrix(a=192.8595, d=27.1283, th=122.9319):
    """
    Using the RA (a) DEC (d) of Galactic north, and theta, generate matrix
    Default values are from J2000

    tested
    """
    try:
        assert a.unit == 'deg'
    except (AttributeError, AssertionError):
        a = a * u.deg
        d = d * u.deg
        th = th * u.deg
    first_t = np.array([
        [np.cos(a),  np.sin(a), 0],
        [np.sin(a), -np.cos(a), 0],
        [        0,          0, 1] 
    ])

    second_t = np.array([
        [-np.sin(d),  0, np.cos(d)],
        [         0, -1,         0],
        [np.cos(d),   0, np.sin(d)]
    ])
    third_t = np.array([
        [np.cos(th),  np.sin(th), 0],
        [np.sin(th), -np.cos(th), 0],
        [         0,           0, 1]
    ])
    return np.dot(third_t, np.dot(second_t, first_t))

def calcGCToEQMatrix(a=192.8595, d=27.1283, th=122.9319):
    """
    Tested
    """
    return np.linalg.inv(calcEQToGCMatrix(a, d, th))

def convertAnglesToCartesian(theta, phi):
    """
    theta   : angle (as astropy degrees) about the north pole (longitude, RA)
    phi : angle (as astropy degrees) from the plane (lattitude, dec))

    Tested
    """
    try:
        assert theta.unit == 'deg'
    except (AttributeError, AssertionError):
        theta = theta * u.deg
        phi = phi * u.deg
    x = np.cos(phi)*np.cos(theta)
    y = np.cos(phi)*np.sin(theta)
    z = np.sin(phi)
    return np.array((x,y,z))

def convertCartesianToAngles(x,y,z):
    """Tested"""
    #normalise values:
    dist = np.sqrt(x**2 + y**2 + z**2)
    phi = (np.arcsin(z/dist)*u.rad).to('deg')
    theta = np.mod((np.arctan2(y/dist,x/dist)*u.rad).to('deg'), 360*u.deg)
    return theta, phi 

def convertEquatorialToGalactic(theta, phi, value=True):
    """Tested"""
    logging.debug("Converting eq ({}, {}) to gc: ".format(theta, phi))
    try:
        assert theta.unit == 'deg'
    except (AttributeError, AssertionError):
        theta = theta * u.deg
        phi = phi * u.deg
    cart_eq = convertAnglesToCartesian(theta, phi)
    logging.debug("Cartesian eq coords: {}".format(cart_eq))
    cart_gc = np.dot(modern_eq_to_gc, cart_eq)
    logging.debug("Cartesian gc coords: {}".format(cart_gc))
    pos_gc = convertCartesianToAngles(*cart_gc)
    if value:
        return [a.value for a in pos_gc]
    else:
        return pos_gc

def convertGalacticToEquatorial(theta, phi, value=True):
    logging.debug("Converting gc ({}, {}) to eq:".format(theta, phi))
    try:
        assert theta.unit == 'deg'
    except (AttributeError, AssertionError):
        theta = theta * u.deg
        phi = phi * u.deg
    cart_gc = convertAnglesToCartesian(theta, phi)
    logging.debug("Cartesian eq coords: {}".format(cart_gc))
    cart_eq = np.dot(modern_gc_to_eq, cart_gc)
    logging.debug("Cartesian gc coords: {}".format(cart_eq))
    pos_eq = convertCartesianToAngles(*cart_eq)
    if value:
        return [a.value for a in pos_eq]
    else:
        return pos_eq

def calcPMCoordinateMatrix(a, d):
    """
    Generate a coordinate matrix for calculating proper motions
    """
    try:
        assert a.unit == 'deg'
    except (AttributeError, AssertionError):
        a = a * u.deg
        d = d * u.deg

    first_t = np.array([
        [ np.cos(d),  0, -np.sin(d)],
        [         0, -1,          0],
        [-np.sin(d),  0, -np.cos(d)]
    ])
    second_t = np.array([
        [np.cos(a),  np.sin(a), 0],
        [np.sin(a), -np.cos(a), 0],
        [        0,         0, -1],
    ])
    return np.dot(second_t, first_t)

def convertPMToSpaceVelocity(a, d, pi, mu_a, mu_d, rv):
    """
    Convert proper motions to space velocities

    Paramters
    ---------
    a : (deg) right ascension in equatorial coordinates
    d : (deg) declination in equatorial coordinates
    pi : (arcsec) parallax
    mu_a : (arcsec/yr) proper motion in right ascension
    mu_d : (arcsec/yr) proper motion in declination
    rv : (km/s) radial velocity

    Returns
    -------
    UVW : [3] array
    """
    try:
        assert a.unit == 'deg'
    except (AttributeError, AssertionError):
        a = a * u.deg
        d = d * u.deg

    B = np.dot(
        calcEQToGCMatrix(),
        calcPMCoordinateMatrix(a, d),
    )
    K = 4.74057 #(km/s) / (1AU/yr)
    astr_vels = np.array([
        rv,
        K * mu_a / pi,
        K * mu_d / pi
    ])
    space_vels = np.dot(B, astr_vels)
    return space_vels


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    pos_gc = (10,-30)
    pos_gc = (0,90)
    xyz_gc = convertAnglesToCartesian(*pos_gc)
    logging.info("Pos_gc is {} in spherical coords".format(pos_gc))
    logging.info("Pos_gc is {} in cartesian coords".format(xyz_gc))
    pos_eq = convertGalacticToEquatorial(*pos_gc)
    xyz_eq = convertAnglesToCartesian(*pos_eq)
    logging.info("Pos_eq is {} in spherical coords".format(pos_eq))
    logging.info("Pos_eq is {} in cartesian coords".format(xyz_eq))
    pos_gc2 = convertEquatorialToGalactic(*pos_eq)
    xyz_gc2 = convertAnglesToCartesian(*pos_gc2)
    logging.info("Pos_gc2 is {} in spherical coords".format(pos_gc2))
    logging.info("Pos_gc2 is {} in cartesian coords".format(xyz_gc2))

    logging.info("Trialing BPic")
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


    astr_bp = [ra_deg, dec_deg, pi*1e-3, mu_ra*1e-3, mu_dec*1e-3, vlos]
    # [86.82, -51.066, 51.44, 4.65, 83.1, 20.0]

    xyzuvw_bp = np.array([-3.4, -16.4, -9.9, -11.0, -16.0, -9.1])
    XYZUVWSOLARNOW = np.array([0., 0., 0.025, 11.1, 12.24, 7.25])

    assert np.allclose(
        xyzuvw_bp[3:],
        convertPMToSpaceVelocity(*astr_bp),
        atol=0.1
    )
