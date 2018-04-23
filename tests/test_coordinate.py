"""
Testing the coordinates.py moduel - a utility module for converting
between various coordinate systems
"""
import numpy as np
import sys

import astropy.units as u
#from astropy.coordinates import SkyCoord

#from galpy.util import bovy_coords
import logging

sys.path.insert(0,'..')

import chronostar.coordinate as cc

XYZUVWSOLARNOW = np.array([0., 0., 0.025, 11.1, 12.24, 7.25])

def test_calcEQToGCMatrix():
    """
    Check the implementation of Johnson and Soderblom 1987
    """
    old_a_ngp = 192.25 * u.degree
    old_d_ngp = 27.4 * u.degree
    old_th = 123 * u.degree

    js1987 = np.array([
        [-0.06699, -0.87276, -0.48354],
        [0.49273, -0.45035, 0.74458],
        [-0.86760, -0.18837, 0.46020],
    ])

    assert np.allclose(
        js1987, cc.calcEQToGCMatrix(old_a_ngp,old_d_ngp,old_th),
        rtol=1e-4
    )

    js1987_inv = cc.calcGCToEQMatrix(old_a_ngp,old_d_ngp,old_th)
    assert np.allclose(
        js1987_inv, np.linalg.inv(js1987), rtol=1e-4
    )

    assert np.allclose(
        np.dot(
            cc.calcEQToGCMatrix(old_a_ngp,old_d_ngp,old_th),
            cc.calcGCToEQMatrix(old_a_ngp,old_d_ngp,old_th),
        ), np.eye(3),
        rtol=1e-4
    )


def test_CartesianAngleConversions():
    """Take position angles to cartesian vectors (on unit sphere) and back"""
    sphere_pos_list = [
        (0,0),
        (45,0),
        (315,0),
        (0,45),
        (0,90),
    ]

    cart_list = [
        (1,0,0),
        (np.sqrt(0.5), np.sqrt(0.5), 0),
        (np.sqrt(0.5), -np.sqrt(0.5), 0),
        (np.sqrt(0.5), 0, np.sqrt(0.5)),
        (0, 0, 1)
    ]

    for sphere_pos, cart in zip(sphere_pos_list, cart_list):
        cart = cc.convertAnglesToCartesian(*sphere_pos)
        sphere_pos2 = cc.convertCartesianToAngles(*cart)
        assert np.allclose(sphere_pos,
                           [c.value for c in sphere_pos2])

def test_convertEquatorialToGalactic():
    """Take equatorial coords to galactic and back"""
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    pos_ncp_list = [
        (20,85),
        (30,20),
        (260, -30),
        (100, -60)
    ]
    for pos_ncp in pos_ncp_list:
        pos_ncp_gc = cc.convertEquatorialToGalactic(*pos_ncp, value=False)
        pos_ncp2 = cc.convertGalacticToEquatorial(*pos_ncp_gc, value=True)
        assert np.allclose(pos_ncp, pos_ncp2)

def test_famousPositions():
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    # galactic north
    gnp_eq = (192.8595, 27.1283)
    gnp_gc = (0,90)

    # testing gnp_eq --> gnp_gc is tricky, since latitude=90 is degenerative
    # w.r.t longitude
    assert np.allclose(
        cc.convertEquatorialToGalactic(*gnp_eq)[1],
        gnp_gc[1], rtol=1e-4
    )
    assert np.allclose(
        cc.convertGalacticToEquatorial(*gnp_gc),
        gnp_eq, rtol=1e-4
    )

    # beta pic
    bp_eq = [86.82125, -51.0664]
    bp_gc = [258.3638, -30.6117]
    assert np.allclose(cc.convertEquatorialToGalactic(*bp_eq), bp_gc)
    assert np.allclose(cc.convertGalacticToEquatorial(*bp_gc), bp_eq)

def test_convertPMToSpaceVelocity():
    astr_bp = [ # astrometry from wikiepdia
        86.82125, #deg
        -51.0664, #deg
        0.05144,  #as
        0.00465,  #as/yr
        0.0831,   #as/yr
        20.0      #km/s
    ]

    # kinematics (from Mamajek & Bell 2014)
    xyzuvw_bp = np.array([-3.4, -16.4, -9.9, -11.0, -16.0, -9.1])
    uvw = cc.convertPMToHelioSpaceVelocity(*astr_bp)
    assert np.allclose(uvw, xyzuvw_bp[3:], rtol=1e-2)

    pos_uvw_bp = np.hstack((astr_bp[:3], xyzuvw_bp[3:]))
    logging.debug("Input: {}".format(pos_uvw_bp))
    logging.debug("Input: {} {} {} {} {} {}".format(*pos_uvw_bp))
    pms_bp = cc.convertHelioSpaceVelocityToPM(*pos_uvw_bp)
    assert np.allclose(pms_bp, astr_bp[3:], rtol=1e-1)

def test_convertHelioXYZUVWToAstrometry():
    astr_bp = [ # astrometry from wikiepdia
        86.82125, #deg
        -51.0664, #deg
        0.05144,  #as
        0.00465,  #as/yr
        0.0831,   #as/yr
        20.0      #km/s
    ]
    xyzuvw_bp_helio = np.array([-3.4, -16.4, -9.9, -11.0, -16.0, -9.1])
    calculated_astr_bp = cc.convertHelioXYZUVWToAstrometry(xyzuvw_bp_helio)
    assert np.allclose(astr_bp, calculated_astr_bp, rtol=1e-2)

    calculated_xyzuvw_bp_helio = cc.convertAstrometryToHelioXYZUVW(
        *astr_bp
    )
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    assert np.allclose(calculated_xyzuvw_bp_helio, xyzuvw_bp_helio, rtol=1e-2)

def test_convertLSRXYZUVWToAstrometry():
    astr_bp = [ # astrometry from wikiepdia
        86.82125, #deg
        -51.0664, #deg
        51.44,  #mas
        4.65,  #mas/yr
        83.1,   #mas/yr
        20.0      #km/s
    ]
    xyzuvw_bp_helio = np.array([-3.4, -16.4, -9.9, -11.0, -16.0, -9.1])
    xyzuvw_bp_lsr =  xyzuvw_bp_helio + XYZUVWSOLARNOW

    calculated_astr_bp = cc.convertLSRXYZUVWToAstrometry(xyzuvw_bp_lsr)
    assert np.allclose(astr_bp, calculated_astr_bp, rtol=1e-2)

    calculated_xyzuvw_bp_lsr = cc.convertAstrometryToLSRXYZUVW(
        *astr_bp
    )
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    assert np.allclose(calculated_xyzuvw_bp_lsr, xyzuvw_bp_lsr, rtol=0.15)

    calculated_astr_bp2 = cc.convertLSRXYZUVWToAstrometry(xyzuvw_bp_lsr)
    calculated_xyzuvw_bp_lsr2 = cc.convertAstrometryToLSRXYZUVW(
        *calculated_astr_bp2
    )
    assert np.allclose(calculated_xyzuvw_bp_lsr2, xyzuvw_bp_lsr)
