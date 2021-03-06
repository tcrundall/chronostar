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
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.potential import MiyamotoNagaiPotential

sys.path.insert(0, '..')
import chronostar.traceorbit as torb

LOGGINGLEVEL = logging.DEBUG

def test_traceforwardThenBack():
    """Check that tracing a point forward then back for the same time step
    returns initial position
    """
    return
    ABS_TOLERANCE = 1e-3
    xyzuvws = np.array([
        [0.,0.,25.,0.,0.,0.],
        # [10.,0.,-50.,0.,0.,0.],
        # [0.,0.,0.,10.,25.,30.,],
    ])
    age = 100.
    times = np.linspace(0,100,1001)
    for xyzuvw_start in xyzuvws:
        galpy_start = None
        xyzuvw_end = torb.trace_cartesian_orbit(xyzuvw_start,
                                                times=age,
                                                single_age=True,
                                                )
        xyzuvw_start_again = torb.trace_cartesian_orbit(xyzuvw_end,
                                                        times=-age,
                                                        single_age=True,
                                                        )
        assert np.allclose(xyzuvw_start, xyzuvw_start_again,
                           atol=ABS_TOLERANCE)


def test_galpy_stationary_conversions():
    """Check if gaply conversions behave as expected where everything
    is at time 0"""

    # Test LSR
    lsr_chron = np.zeros(6)
    lsr_galpy = np.array([1.,0,1,0,0,0])

    assert np.allclose(lsr_chron,
                       torb.convert_galpycoords2cart(lsr_galpy, ts=0.))
    assert np.allclose(lsr_galpy,
                       torb.convert_cart2galpycoords(lsr_chron, ts=0.))

    # Test galactic centre
    gc_chron = np.array([8000.,0,0,0,-220.,0,])
    gc_galpy = np.ones(6) * 1e-15

    assert np.allclose(gc_chron,
                       torb.convert_galpycoords2cart(gc_galpy, ts=0.))
    assert np.allclose(gc_galpy,
                       torb.convert_cart2galpycoords(gc_chron, ts=0.))

    # Test simple, off origin point
    off_chron = np.array([4000, 8000.*np.sqrt(3)/2, 0,
                          np.sin(np.pi/3)*220.,
                          -np.cos(np.pi/3)*220.,
                          0])
    off_galpy = np.array([1.,0,1,0,0,np.pi/3.])

    assert np.allclose(off_galpy,
                       torb.convert_cart2galpycoords(off_chron, ts=0.))
    assert np.allclose(off_chron,
                       torb.convert_galpycoords2cart(off_galpy, ts=0.))


    # Test random positions
    SPREAD = 100000
    NSAMPLES = int(1e6)
    many_pos_chron = (np.random.rand(NSAMPLES,6) - 0.5) * SPREAD # uniform between -10 and 10
    many_pos_galpy = torb.convert_cart2galpycoords(many_pos_chron, ts=0.)

    assert np.allclose(many_pos_chron,
                       torb.convert_galpycoords2cart(many_pos_galpy, ts=0.),
                       atol=1e-2)

def test_galpy_moving_conversions():
    """Check if gaply conversions behave as expected where time
    is allowed to vary."""
    lsr_chron = np.zeros(6)
    lsr_galpy = np.array([1.,0,1,0,0,0])
    # Incorporate positive time into lsr position checks
    NSTEPS = 10
    galpy_times = np.linspace(0., 2*np.pi, NSTEPS)
    lsrs_chron = np.repeat(lsr_chron, NSTEPS).reshape(6,-1).T
    lsrs_galpy = np.repeat(lsr_galpy, NSTEPS).reshape(6,-1).T
    lsrs_galpy[:,-1] = galpy_times
    chron_times = torb.convert_bovytime2myr(galpy_times)

    assert np.allclose(
        lsrs_chron,
        torb.convert_galpycoords2cart(lsrs_galpy, ts=galpy_times))
    assert np.allclose(
        lsrs_galpy,
        torb.convert_cart2galpycoords(lsrs_chron, ts=chron_times)
    )

    # Incorporate negative time into lsr position checks
    galpy_times = np.linspace(0., -2*np.pi, NSTEPS)
    lsrs_chron = np.repeat(lsr_chron, NSTEPS).reshape(6,-1).T
    lsrs_galpy = np.repeat(lsr_galpy, NSTEPS).reshape(6,-1).T
    lsrs_galpy[:,-1] = galpy_times
    chron_times = torb.convert_bovytime2myr(galpy_times)

    assert np.allclose(
        lsrs_chron,
        torb.convert_galpycoords2cart(lsrs_galpy, ts=galpy_times))
    assert np.allclose(
        lsrs_galpy,
        torb.convert_cart2galpycoords(lsrs_chron, ts=chron_times)
    )

    # Test random positions with random times
    SPREAD = int(1e4) # pc
    NSAMPLES = 100
    many_pos_chron = (np.random.rand(NSAMPLES,6) - 0.5) * SPREAD # uniform between -10 and 10
    many_chron_times = np.random.rand(NSAMPLES) * 100 #Myr
    many_pos_galpy = torb.convert_cart2galpycoords(
        many_pos_chron, ts=many_chron_times
    )
    many_galpy_times = torb.convert_myr2bovytime(many_chron_times)

    for i in range(NSAMPLES):
        assert np.allclose(many_pos_chron[i],
                           torb.convert_galpycoords2cart(
                               many_pos_galpy[i], ts=many_galpy_times[i]
                           ),
                           atol=1e-2)

def test_careful_traceback_and_forward():
    """Step by step, project orbit forward, then backward"""
    bovy_times = np.array([0., np.pi/3.])
    chron_times = torb.convert_bovytime2myr(bovy_times)

    init_pos_chron = np.array([
        4000, 8000.*np.sqrt(3)/2, 0,
        np.sin(np.pi/3)*220.,
        -np.cos(np.pi/3)*220.,
        0
    ])
    init_pos_galpy = torb.convert_cart2galpycoords(init_pos_chron, ts=0.)

    assert np.allclose(np.array([1.,0,1,0,0,np.pi/3.]),
                       init_pos_galpy)

    o = Orbit(vxvv=init_pos_galpy, ro=8., vo=220.)
    o.integrate(bovy_times, MWPotential2014, method='odeint')

    orbit_galpy = o.getOrbit()
    assert np.allclose(init_pos_galpy, orbit_galpy[0])
    assert np.allclose(init_pos_galpy
                       + np.array([0.,0.,0.,0.,0.,bovy_times[-1]]),
                       orbit_galpy[-1])

    orbit_chron = torb.convert_galpycoords2cart(orbit_galpy,
                                                ts=bovy_times)
    assert np.allclose(init_pos_chron, orbit_chron[0])
    assert np.allclose(init_pos_chron,
                       orbit_chron[-1])

    # Setup for backwards time integration
    # Currently at time of PI/3
    back_init_pos_chron = orbit_chron[-1]
    back_init_pos_galpy = torb.convert_cart2galpycoords(
        back_init_pos_chron,
        bovy_times=bovy_times[-1],
    )

    assert np.allclose(back_init_pos_galpy,
                       torb.convert_cart2galpycoords(
                           back_init_pos_chron,
                           bovy_times=bovy_times[-1]
                       ))

    back_o = Orbit(vxvv=back_init_pos_galpy, ro=8., vo=220.)
    back_o.integrate(-1*bovy_times, MWPotential2014, method='odeint')

    back_orbit_galpy = back_o.getOrbit()

    assert np.allclose(back_init_pos_galpy, back_orbit_galpy[0])
    assert np.allclose(back_init_pos_galpy
                       - np.array([0.,0.,0.,0.,0.,bovy_times[-1]]),
                       back_orbit_galpy[-1])
    assert np.allclose(init_pos_galpy, back_orbit_galpy[-1])

    back_orbit_chron = torb.convert_galpycoords2cart(
        back_orbit_galpy,
        ts=bovy_times[::-1],
    )

    assert np.allclose(init_pos_chron, back_orbit_chron[-1])



def test_traceback_and_forward():
    """The test that shows things are broken"""
    time = 10. #Myr
    times = np.array([0., 10.])
    init_pos_chron = np.array([10.,0.,30.,0.,0.,0.])
    # init_pos_chron = np.zeros(6)
    init_pos_chron = np.array([4000, 8000.*np.sqrt(3)/2, 0,
                          np.sin(np.pi/3)*220.,
                          -np.cos(np.pi/3)*220.,
                          0])
    final_pos_chron = torb.trace_cartesian_orbit(
        init_pos_chron, times=times, single_age=False)[-1]
    final_pos_chron = torb.traceforward_from_now(
        init_pos_chron, time=time,
    )
    assert np.allclose(
        init_pos_chron,
        torb.traceback_to_now(
            final_pos_chron, time,
            ),
        atol=1e-5
    )

def test_multi_traceback_and_forward():
    np.random.seed(0)
    NPOSITIONS = 10
    init_positions = np.random.rand(NPOSITIONS, 6) * 20 - 10
    time_spans = np.random.rand(NPOSITIONS) * 30
    for pos, time in zip(init_positions, time_spans):
        final_pos = torb.traceforward_from_now(pos, time)
        init_pos = torb.traceback_to_now(final_pos, time)
        assert np.allclose(pos, init_pos, atol=1e-3)


    for pos, time in zip(init_positions, time_spans):
        print('time: {}'.format(time))
        final_pos = torb.traceback_from_now(pos, time)
        init_pos = torb.traceforward_to_now(final_pos, time)
        assert np.allclose(pos, init_pos, atol=1e-3)


def test_interval_tracing():
    np.random.seed(0)
    start = np.random.rand(6) * 20 - 10

    time_steps = [3.,-10.,-3.,10]

    current_pos = start
    for time_step in time_steps:
         current_pos = torb.base_trace_cartesian_orbit(
             current_pos,
             end_time = time_step,
         )
    assert np.allclose(start, current_pos, atol=1e-3)


def test_interval_tracing_orig():
    np.random.seed(0)
    start = np.random.rand(6) * 20 - 10

    time_steps = [3.,-10.,-3.,10]

    current_pos = start
    for time_step in time_steps:
         current_pos = torb.trace_cartesian_orbit(
             current_pos,
             times=time_step,
             single_age=True,
         )
    assert np.allclose(start, current_pos, 1e-3)