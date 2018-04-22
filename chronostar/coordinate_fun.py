import logging
import numpy as np
import astropy.units as u

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

def calcGCToEQMatrix(a, d, th):
    """
    Using the RA (a) DEC (d) of Galactic north, and theta, generate matrix
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

def calcEQToGCMatrix(a, d, th):
    return np.linalg.inv(calcGCToEQMatrix(a, d, th))

def convertAnglesToCartesian(theta, phi):
    """
    theta   : angle (as astropy degrees) about the north pole (longitude, RA)
    phi : angle (as astropy degrees) from the plane (lattitude, dec))
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
    phi = (np.arcsin(z)*u.rad).to('deg')
    theta = (np.arctan2(y,x)*u.rad).to('deg')
    return theta, phi 

def convertEquatorialToGalactic(theta, phi):
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
    return convertCartesianToAngles(*cart_gc)

def convertGalacticToEquatorial(theta, phi):
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
    return convertCartesianToAngles(*cart_eq)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    pos_gc = (10,-30)
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
    
