"""
Tests traceorbit module. This module makes great use of Galpy so for
convenience a brief summary of Galpy coordinates is provided:

    Galpy coordinates are [R, vR, vT, z, vz, phi]
    By default, positions are scaled by LSR distance from galactic centre,
    ro=8kpc, and velocities scaled by the LSR circular velocity,
    vo = 220km/s. Time is scaled such that after 1 time unit has passed,
    the LSR has travelled 1 radian about the galactic centre. The values are
    returned in a [ntimes, 6]

    array:
        R : galactic radial distance /ro
        vR : galactic radial velocity /vo
        vT : circular velocity /vo
        z  : vertical distance from plane / ro
        vz : vertical velocity / vo
        phi : angle about the galaxy (anticlockwise from LSR's location at
             t=0)
    For example, the LSR at t=0.0 and t=1.0 as values:
    [1., 0., 1., 0., 0., 0.]
    [1., 0., 1., 0., 0., 1.]
"""

import logging
import numpy as np
import sys

sys.path.insert(0, '..')
import chronostar.traceorbit as torb

LOGGINGLEVEL = logging.DEBUG

def test_LSR():
    """
    Check that LSR remains constant in our frame of reference.

    Since our frame of reference is **centred** on the LSR, then the LSR
    should remain at the origin.
    """
    xyzuvw_lsr = [0.,0.,0.,0.,0.,0.]
    times = np.linspace(0,100,101)

    xyzuvws = torb.traceOrbitXYZUVW(xyzuvw_lsr, times, single_age=False)
    assert np.allclose(xyzuvws[0,:5],xyzuvws[-1,:5])

def test_rotatedLSR():
    """
    Check that LSRs with different azimuthal positions also remain constant
    """
    rot_lsr_gp_coords = np.array([1., 0., 1., 0., 0., np.pi])
    xyzuvw_rot_lsr = torb.convertGalpyCoordsToXYZUVW(rot_lsr_gp_coords)
    times = np.linspace(0,100,101)
    xyzuvws = torb.traceOrbitXYZUVW(xyzuvw_rot_lsr, times, single_age=False)

    # On a circular orbit, same radius as LSR, so shouldn't vary at all
    assert np.allclose(xyzuvws[0,:5],xyzuvws[-1,:5])

    # Should be initialised on opposite side of galaxy (X = 16kpc)
    assert np.allclose(xyzuvws[0,0], 16000.)

def test_singleTime():
    """Test usage where we only provide the desired age, and not an array

    Good demo of how to traceback 2 stars forward through time, either
    with an array of time steps, or a single age
    """
    xyzuvw_1 = [0.,0.,25.,0.,0.,0.]
    xyzuvw_2 = [0.,0.,0.,0.,-10.,0.]
    xyzuvws = np.vstack((xyzuvw_1, xyzuvw_2))
    age = 10.
    times = np.linspace(0., age, 2)

    # get position for each time in times
    xyzuvws_both = torb.traceManyOrbitXYZUVW(xyzuvws, times, single_age=False)

    # get position for *age* only
    xyzuvws_now = torb.traceManyOrbitXYZUVW(xyzuvws, age, single_age=True)
    assert np.allclose(xyzuvws_both[:,1], xyzuvws_now)

    xyzuvw_both = torb.traceOrbitXYZUVW(xyzuvws[0], times, single_age=False)
    xyzuvw_now = torb.traceOrbitXYZUVW(xyzuvws[0], age, single_age=True)
    assert np.allclose(xyzuvw_both[1], xyzuvw_now)


def test_traceforwardThenBack():
    """Check that tracing a point forward then back for the same time step
    returns initial position
    """
    ABS_TOLERANCE = 1e-3
    xyzuvws = np.array([
        [0.,0.,25.,0.,0.,0.],
        [10.,0.,-50.,0.,0.,0.],
        [0.,0.,0.,10.,25.,30.,],
    ])
    age = 100.
    times = np.linspace(0,100,1001)
    for xyzuvw_start in xyzuvws:
        xyzuvw_end = torb.traceOrbitXYZUVW(xyzuvw_start,
                                           times=age,
                                           single_age=True,
                                           )
        xyzuvw_start_again = torb.traceOrbitXYZUVW(xyzuvw_end,
                                                   times=-age,
                                                   single_age=True,
                                                   )
        assert np.allclose(xyzuvw_start, xyzuvw_start_again,
                           atol=ABS_TOLERANCE)
