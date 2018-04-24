import logging
import numpy as np
import astropy.units as un

a_o = 192.8595 * un.degree
b_ncp = d_o = 27.1283 * un.degree
l_ncp = l_o = 122.9319 * un.degree

old_a_ngp = 192.25 * un.degree
old_d_ngp = 27.4 * un.degree
old_th = 123 * un.degree

a_ngp = 192.25 * un.degree
d_ngp = 27.4 * un.degree

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
        a = a * un.deg
        d = d * un.deg
        th = th * un.deg
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


def convertAnglesToCartesian(theta, phi, radius=1.0):
    """
    theta   : angle (as astropy degrees) about the north pole (longitude, RA)
    phi : angle (as astropy degrees) from the plane (lattitude, dec))

    Tested
    """
    try:
        assert theta.unit == 'deg'
    except (AttributeError, AssertionError):
        theta = theta * un.deg
        phi = phi * un.deg
    x = radius * np.cos(phi)*np.cos(theta)
    y = radius * np.cos(phi)*np.sin(theta)
    z = radius * np.sin(phi)
    return np.array((x,y,z))


def convertCartesianToAngles(x,y,z,return_dist=False, value=False):
    """Tested"""
    #normalise values:
    dist = np.sqrt(x**2 + y**2 + z**2)
    phi = (np.arcsin(z/dist)*un.rad).to('deg')
    theta = np.mod((np.arctan2(y/dist,x/dist)*un.rad).to('deg'), 360*un.deg)
    if value:
        phi = phi.value
        theta = theta.value
    if return_dist:
        return theta, phi, dist
    else:
        return theta, phi


def convertEquatorialToGalactic(theta, phi, value=True):
    """Tested"""
    logging.debug("Converting eq ({}, {}) to gc: ".format(theta, phi))
    try:
        assert theta.unit == 'deg'
    except (AttributeError, AssertionError):
        theta = theta * un.deg
        phi = phi * un.deg
    cart_eq = convertAnglesToCartesian(theta, phi)
    logging.debug("Cartesian eq coords: {}".format(cart_eq))
    eq_to_gc = calcEQToGCMatrix()
    cart_gc = np.dot(eq_to_gc, cart_eq)
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
        theta = theta * un.deg
        phi = phi * un.deg
    cart_gc = convertAnglesToCartesian(theta, phi)
    logging.debug("Cartesian eq coords: {}".format(cart_gc))
    gc_to_eq = calcGCToEQMatrix()
    cart_eq = np.dot(gc_to_eq, cart_gc)
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
        a = a * un.deg
        d = d * un.deg

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


def convertPMToHelioSpaceVelocity(a, d, pi, mu_a, mu_d, rv):
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
        a = a * un.deg
        d = d * un.deg

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


def convertHelioSpaceVelocityToPM(a, d, pi, u, v, w):
    """Take the position and space velocities, return proper motions and rv

    Paramters
    ---------
    a : (deg) right ascension
    d : (deg) declination
    pi : (as) parallax
    u : (km/s) heliocentric velocity towards galactic centre
    v : (km/s) heliocentric velocity towards in direction of orbit
    w : (km/s) heliocentric velocity towards galactic north

    Returns
    -------
    mu_a : (as/yr) proper motion in right ascension
    mu_d : (as/yr) proper motion in declination
    rv : (km/s) line of sight velocity
    """
    try:
        assert a.unit == 'deg'
    except (AttributeError, AssertionError):
        logging.debug("a is {}".format(a))
        logging.debug("u is {}".format(u))
        a = a * un.deg
        d = d * un.deg

    logging.debug("Parallax is {} as which is a distance of {} pc".format(
        pi, 1./pi
    ))

    space_vels = np.array([u,v,w])

    B_inv = np.linalg.inv(np.dot(
        calcEQToGCMatrix(),
        calcPMCoordinateMatrix(a,d)
    ))
    sky_vels = np.dot(B_inv, space_vels) # now in km/s
    K = 4.74057 #(km/s) / (AU/yr)
    rv = sky_vels[0]
    mu_a = pi * sky_vels[1] / K
    mu_d = pi * sky_vels[2] / K
    return mu_a, mu_d, rv


def convertHelioXYZUVWToAstrometry(xyzuvw_helio):
    """
    Takes as input heliocentric XYZUVW values, returns astrometry

    Parameters
    ----------
    xyzuvw_helio : (pc, pc, pc, km/s, km/s, km/s) array
        The position and velocity of a star in a right handed cartesian system
        centred on the sun

    Returns
    -------
    a : (deg) right ascention
    d : (deg) declination
    pi : (as) parallax
    mu_a : (as/yr) proper motion in right ascension
    mu_d : (as/yr) proper motion in declination
    rv : (km/s) line of sight velocity
    """
    x, y, z, u, v, w = xyzuvw_helio
    l, b, dist = convertCartesianToAngles(x,y,z,return_dist=True)
    logging.debug("Distance is {} pc".format(dist))
    a, d = convertGalacticToEquatorial(l, b)
    pi = 1./dist
    mu_a, mu_d, rv = convertHelioSpaceVelocityToPM(a, d, pi, u, v, w)
    return a, d, pi, mu_a, mu_d, rv


def convertAstrometryToHelioXYZUVW(a, d, pi, mu_a, mu_d, rv):
    """
    Converts astrometry to heliocentric XYZUVW values

    Parameters
    ----------
    a : (deg) right ascention
    d : (deg) declination
    pi : (as) parallax
    mu_a : (as/yr) proper motion in right ascension
    mu_d : (as/yr) proper motion in declination
    rv : (km/s) line of sight velocity
    """
    logging.debug("Input:\nra {}\ndec {}\nparallax {}\nmu_ra {}\nmu_de {}\n"
                  "rv {}".format(a, d, pi, mu_a, mu_d, rv))
    dist = 1/pi #pc
    l, b = convertEquatorialToGalactic(a, d)
    x, y, z = convertAnglesToCartesian(l, b, radius=dist)
    u, v, w = convertPMToHelioSpaceVelocity(a, d, pi, mu_a, mu_d, rv)
    xyzuvw_helio = np.array([x,y,z,u,v,w])
    logging.debug("XYZUVW heliocentric is : {}".format(xyzuvw_helio))
    return xyzuvw_helio


def convertLSRToHelio(xyzuvw_lsr, kpc=False):
    """Assumes position is in pc unless stated otherwise"""
    XYZUVWSOLARNOW = np.array([0., 0., 25., 11.1, 12.24, 7.25])
    if kpc:
        xyzuvw_lsr = np.copy(xyzuvw_lsr)
        xyzuvw_lsr[:3] *= 1e3
        res = (xyzuvw_lsr - XYZUVWSOLARNOW)
        res[:3] *= 1e-3
        return res

    return xyzuvw_lsr - XYZUVWSOLARNOW


def convertHelioToLSR(xyzuvw_helio, kpc=False):
    """Assumes position is in pc unless stated otherwise"""
    XYZUVWSOLARNOW = np.array([0., 0., 25., 11.1, 12.24, 7.25])
    if kpc:
        xyzuvw_lsr = np.copy(xyzuvw_helio)
        xyzuvw_lsr[:3] *= 1e3
        res = (xyzuvw_helio + XYZUVWSOLARNOW)
        res[:3] *= 1e-3
        return res

    return xyzuvw_helio + XYZUVWSOLARNOW


#def convertAstrometryToLSRXYZUVW(a, d, pi, mu_a, mu_d, rv, mas=True):
def convertAstrometryToLSRXYZUVW(astro, mas=True):
    """
    Take a point straight from a catalogue, return it as XYZUVW

    This function takes astrometry in conventional units, and converts them
    into internal units for convenience.

    Parameters
    ----------
    a : (deg) right ascention
    d : (deg) declination
    pi : (mas) parallax
    mu_a : (mas/yr) proper motion in right ascension
    mu_d : (mas/yr) proper motion in declination
    rv : (km/s) line of sight velocity

    mas : Boolean {True}
        set if input parallax and proper motions are in mas

    Returns
    -------
    XYZUVW : (pc, pc, pc, km/s, km/s, km/s)
    """
    astro = np.copy(astro)
    # convert to as for internal use
    if mas:
        astro[2:5] *= 1e-3
    logging.debug("Input (after conversion) is: {}".format(astro))
    xyzuvw_helio = convertAstrometryToHelioXYZUVW(*astro)
    logging.debug("Heliocentric XYZUVW is : {}".format(xyzuvw_helio))
    xyzuvw_lsr = convertHelioToLSR(xyzuvw_helio)

    logging.debug("LSR XYZUVW (pc) is : {}".format(xyzuvw_lsr))
    return xyzuvw_lsr

def convertManyAstrometryToLSRXYZUVW(astr_arr, mas=True):
    """
    Take a point straight from a catalogue, return it as XYZUVW

    This function takes astrometry in conventional units, and converts them
    into internal units for convenience.

    Parameters
    ----------
    a : (deg) right ascention
    d : (deg) declination
    pi : (mas) parallax
    mu_a : (mas/yr) proper motion in right ascension
    mu_d : (mas/yr) proper motion in declination
    rv : (km/s) line of sight velocity

    mas : Boolean {True}
        set if input parallax and proper motions are in mas
    """
    xyzuvws = np.zeros(astr_arr.shape)
    for i, astr in enumerate(astr_arr):
        xyzuvws[i] = convertAstrometryToLSRXYZUVW(astr, mas=mas)
    return xyzuvws


def convertLSRXYZUVWToAstrometry(xyzuvw_lsr):
    """
    Takes as input heliocentric XYZUVW values, returns astrometry

    Parameters
    ----------
    xyzuvw_lsr : (pc, pc, pc, km/s, km/s, km/s) array
        The position and velocity of a star in a right handed cartesian system
        corotating with and centred on the local standard of rest

    Returns
    -------
    a : (deg) right ascention
    d : (deg) declination
    pi : (mas) parallax
    mu_a : (mas/yr) proper motion in right ascension
    mu_d : (mas/yr) proper motion in declination
    rv : (km/s) line of sight velocity
    """
    xyzuvw_lsr = np.copy(xyzuvw_lsr)

    logging.debug("Input (before conversion) is: {}".format(xyzuvw_lsr))

    xyzuvw_helio = convertLSRToHelio(xyzuvw_lsr)
    logging.debug("xyzuvw_helio is: {}".format(xyzuvw_helio))
    astr = np.array(convertHelioXYZUVWToAstrometry(xyzuvw_helio))
    logging.debug("Astro before conversion is: {}".format(astr))

    # Finally converts angles to mas for external use
    astr[2:5] *= 1e3
    logging.debug("Astro after conversion is: {}".format(astr))
    return astr


def convertManyLSRXYZUVWToAstrometry(xyzuvw_lsrs):
    """
    Takes as input heliocentric XYZUVW values, returns astrometry

    Parameters
    ----------
    xyzuvw_lsr : (pc, pc, pc, km/s, km/s, km/s) array
        The position and velocity of a star in a right handed cartesian system
        corotating with and centred on the local standard of rest

    Returns
    -------
    a : (deg) right ascention
    d : (deg) declination
    pi : (mas) parallax
    mu_a : (mas/yr) proper motion in right ascension
    mu_d : (mas/yr) proper motion in declination
    rv : (km/s) line of sight velocity
    """
    astros = np.zeros(xyzuvw_lsrs.shape)
    for i, xyzuvw_lsr in enumerate(xyzuvw_lsrs):
        astros[i] = convertLSRXYZUVWToAstrometry(xyzuvw_lsr)
    return astros
