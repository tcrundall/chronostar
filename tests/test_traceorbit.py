import logging
import numpy as np
import sys

sys.path.insert(0, '..')
import chronostar.traceorbit as torb

LOGGINGLEVEL = logging.DEBUG

def testLSR():
    logging.basicConfig(level=LOGGINGLEVEL, stream=sys.stdout)
    xyzuvw_lsr = [0.,0.,0.,0.,0.,0.]
    times = np.linspace(0,100,101)

    xyzuvws = torb.traceOrbitXYZUVW(xyzuvw_lsr, times)
    assert np.allclose(xyzuvws[0,:5],xyzuvws[-1,:5])

def testRotatedLSR():
    logging.basicConfig(level=LOGGINGLEVEL, stream=sys.stdout)
    rot_lsr_gp_coords = np.array([1., 0., 1., 0., 0., np.pi])
    xyzuvw_rot_lsr = torb.convertGalpyCoordsToXYZUVW(rot_lsr_gp_coords)
    times = np.linspace(0,100,101)
    xyzuvws = torb.traceOrbitXYZUVW(xyzuvw_rot_lsr, times)

    # On a circular orbit, same radius as LSR, so shouldn't vary at all
    assert np.allclose(xyzuvws[0,:5],xyzuvws[-1,:5])

    # Should be initialised on opposite side of galaxy (X = 16kpc)
    assert np.allclose(xyzuvws[0,0], 16000.)

def testSingleTime():
    """Test usage where we only provide the desired age, and not an array"""
    logging.basicConfig(level=LOGGINGLEVEL, stream=sys.stdout)
    xyzuvw_1 = [0.,0.,25.,0.,0.,0.]
    xyzuvw_2 = [0.,0.,0.,0.,-10.,0.]
    xyzuvws = np.vstack((xyzuvw_1, xyzuvw_2))
    age = 10.
    times = np.linspace(0., age, 2)
    xyzuvws_both = torb.traceManyOrbitXYZUVW(xyzuvws, times)
    xyzuvws_now = torb.traceManyOrbitXYZUVW(xyzuvws, age=age)
    assert np.allclose(xyzuvws_both[:,1], xyzuvws_now)

    xyzuvw_both = torb.traceOrbitXYZUVW(xyzuvws[0], times)
    xyzuvw_now = torb.traceOrbitXYZUVW(xyzuvws[0], age=age)
    assert np.allclose(xyzuvw_both[1], xyzuvw_now)
