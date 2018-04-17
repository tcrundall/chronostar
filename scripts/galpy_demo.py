from galpy.orbit import Orbit
from galpy.potential import MWPotential2014 as mp
from galpy.util import bovy_conversion
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

import pdb

from IPython.core.debugger import Tracer

# GALPY STORES ORBITAL INFO IN GALACTOCENTRIC CYLINDRICAL COORDINATES
# [R, vR, vT, z, vZ, phi]

def observed_to_xyzuvw(obs, ts, lsr_orbit=None):
    """
    Convert six-parameter astrometric solution to XYZUVW orbit.

    Parameters
    ----------
    obs :   [RA (deg), DEC (deg), pi (mas),
             mu_ra (mas/yr), mu_dec (mas/yr), vlos (km/s)]
        Current kinematics
    ts : [ntimes] array
        times (in Gyr) to traceback to

    lsr_orbit : Orbit
        the orbit of the local standard of rest for comparison, if None can
        calculate on the fly

    XYZUVW : [ntimes, 6] array
        The space position and velocities of the star in a co-rotating frame
        centred on the LSR
    """

    Tracer()()

def galpy_coords_to_xyzuvw(data, ts, ro=8., vo=220., rc=True):
    """
    Converts orbits from galactocentric and takes them to XYZUVW

    Data should be raw galpy data (i.e. output from o.getOrbit()).
    The XYZUVW will be a corotating reference frame centred on the LSR
    as defined by the Schoenrich solar motion of
    XYZUVW = 0, 0, 25pc, 11.1 km/s, 12.24 km/s, 7.25 km/s

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
        phi : angle about the galaxy
    ts : [ntimes] float array [galpy time units]
        times used to generate orbit. Ensure the units are in galpy time units
    ro : float
        a conversion factor that takes units from galpy units to
        physical units. If left as default, output will be in kpc
    vo : float
        a conversion factor that takes units form galpy units to
        physical units. If left as default, output will be in km/s
        This is also the circular velocity of a circular orbit with X,Y
        equal to that of the sun.
    rc : boolean
        whether to calculate XYZUVW in a right handed coordinate system
        (X, U positive towards galactic centre)
    """
    phi_lsr = ts
    R, vR, vT, z, vz, phi_s = data.T

    phi = phi_s - phi_lsr

    X = ro * (1. - R * np.cos(phi))
    Y = ro * (- np.sin(phi))
    Z = ro * z

    U = vo * (-vR*np.cos(phi) - vT*np.sin(phi))
    V = vo * ( vT*np.cos(phi) - vR*np.sin(phi) - 1.)
    W = vo * vz
    #pdb.set_trace()

    XYZUVW = np.vstack((X,Y,Z,U,V,W)).T
    return XYZUVW

def demo_lsr_and_sun_cal():
    """
    Litte demo showing how one would calculate the orbit of the LSR and sun
    :return:
    """
    perimeter = 2 * np.pi * 8 * u.kpc
    velocity  = 220 * u.km/ u.s
    # for reference, LSR (at 8 kpc, with V = 220 km/s) should take this long
    # to complete one orbit
    orbit_time = (perimeter / velocity).to("Gyr")

    max_age = 100 * orbit_time / bovy_conversion.time_in_Gyr(220., 8.) # Gyr
    ntimes = 10000
    ts = np.linspace(0,max_age,ntimes)

    # INITIALISING SUN COORDINATES AND ORBIT
    #deg, deg, kpc,  mas/yr, mas/yr, km/s
    ra, dec, dist, mu_ra,  mu_dec, vlos = 0., 0., 0., 0., 0., 0.
    solar_coords = [ra, dec, dist, mu_ra, mu_dec, vlos]
    sun = Orbit(vxvv=solar_coords, radec=True, solarmotion='schoenrich') # should just be the sun's orbit
    sun.integrate(ts,mp,method='odeint')

    # get the orbit [R, vR, vT, z, vz, phi] (pos scaled by ro, vel scaled by vo)
    sun_data = sun.getOrbit()

    # plots the sun's motion with respect to Galactic Centre
    sunR = 8 * sun_data[:,0]
    sunphi = sun_data[:,5]
    sunX = sunR * np.cos(sunphi)
    sunY = sunR * np.sin(sunphi)
    sunZ = 8 * sun_data[:,3]
    plt.clf()
    plt.plot(sunX, sunY)
    plt.savefig('temp_plots/sunXY.png')

    plt.clf()
    plt.plot(sunX, sunZ)
    plt.savefig('temp_plots/sunXZ.png')

    # plot the XY of the sun's motion using galpy's plot function (w.r.t GC)
    plt.clf()
    sun.plot(d1='x', d2='y')
    plt.savefig('temp_plots/galpy_sunXY.png')

    sun.plot(d1='x', d2='z')
    plt.savefig('temp_plots/galpy_sunXZ.png')

    plt.clf()
    sun.plot(d1='R', d2='z')
    plt.savefig('temp_plots/galpy_sunRZ.png')

    #                                                        kpc, km/s
    # INITIALISING THE LSR (at XYZUVW (w.r.t sun) of [0,0,-0.025,0,220,0]
    R, vR, vT, z, vz, phi = 1., 0., 1., 0., 0., 0. # <--- Galpy units
    LSR_coords = [R, vR, vT, z, vz, phi]
    lsr = Orbit(vxvv=LSR_coords, solarmotion='schoenrich', vo=220, ro=8)
    lsr.integrate(ts, mp, method='odeint')

    # plots a perfect circle
    plt.clf()
    lsr.plot(d1='x', d2='y')
    plt.savefig('temp_plots/galpy_lsrXY.png')

    plt.clf()
    lsr.plot(d1='x', d2='z')
    plt.savefig('temp_plots/galpy_lsrXZ.png')

    # Manually reconstructing orbit
    lsr_data = lsr.getOrbit()
    lsrR = 8 * lsr_data[:,0]
    lsrphi = lsr_data[:,5]

    lsrX = lsrR * np.cos(lsrphi)
    lsrY = lsrR * np.sin(lsrphi)
    lsrZ = 8 * lsr_data[:,3]

    plt.clf()
    plt.plot(lsrX, lsrY)
    plt.savefig('temp_plots/lsrXY.png')
    plt.clf()
    plt.plot(lsrX, lsrZ)
    plt.savefig('temp_plots/lsrXZ.png')

    # plotting both sun and lsr
    plt.clf()
    plt.plot(lsrX, lsrY)
    plt.plot(sunX, sunY)
    plt.savefig('temp_plots/combXY.png')
    plt.clf()
    plt.plot(lsrX, lsrZ)
    plt.plot(sunX, sunZ)
    plt.savefig('temp_plots/combXZ.png')

    # Finding sun's path w.r.t the LSR in non-corotating frame
    relsunX = sunX - lsrX
    relsunY = sunY - lsrY
    relsunZ = sunZ - lsrZ

    plt.clf()
    plt.plot(relsunX, relsunY)
    plt.savefig('temp_plots/relsunXY.png')
    plt.clf()
    plt.plot(relsunX, relsunZ)
    plt.savefig('temp_plots/relsunXZ.png')

    # Getting sun's path w.r.t the LSR in cortating frame
    sun_rel_data = sun_data - lsr_data
    sun_relR = 8 * sun_rel_data[:,0]
    sun_relphi = sun_rel_data[:,5]

    sun_relX = sun_relR * np.cos(sun_relphi)
    sun_relY = sun_relR * np.sin(sun_relphi)
    sun_relZ = 8 * sun_rel_data[:,3]

    plt.clf()
    plt.plot(sun_relX, sun_relY)
    plt.savefig('temp_plots/sun_relXY.png')
    plt.clf()
    plt.plot(sun_relX, sun_relZ)
    plt.savefig('temp_plots/sun_relXZ.png')

    # Try and plot LSR and sun in 3D for comparison with
    # relative plot
    plt.clf()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    #z = np.linspace(-2, 2, 100)
    #r = z ** 2 + 1
    #x = r * np.sin(theta)
    #y = r * np.cos(theta)
    ax.plot(sunX, sunY, sunZ, label='sun')
    ax.plot(lsrX, lsrY, lsrZ, label='lsr')
    ax.legend()
    plt.savefig('temp_plots/3D_sun_lsr.png')
    plt.show()
    #galpy_coords_to_xyzuvw(lsr_data)
    print("Max age is {} and max phi is {}... does this make sense?".\
        format(max_age, np.max(lsr_data[:,5])))
    print("Max age is {} and max phi is {}... does this make sense?". \
        format(max_age, np.max(sun_data[:,5])))


if __name__ == '__main__':
#def lsr_nonsense():
    #demo_lsr_and_sun_cal()

    RO = 8.
    VO = 220.
    BOVY_TIME_CONVERSION = bovy_conversion.time_in_Gyr(VO, RO) * 1000 # Myr/bovy_time

    perimeter = 2 * np.pi * 8 * u.kpc
    velocity = 220 * u.km / u.s
    # for reference, LSR (at 8 kpc, with V = 220 km/s) should take this long
    # to complete one orbit
    orbit_time = (perimeter / velocity).to("Myr")

    max_age = orbit_time.value / BOVY_TIME_CONVERSION
    ntimes = 100
    ts = np.linspace(0, max_age, ntimes)

    # demo a star (with vT=220, vR=0, vZ=0, z=0, phi=0.1 pi) staying
    # fixed in our coordinate frame
    R, vR, vT, z, vz, phi = 1., 0., 1., 0., 0., 0.
    LSR_coords = [R, vR, vT, z, vz, phi]
    lsr = Orbit(vxvv=LSR_coords, solarmotion='schoenrich', vo=220, ro=8)
    lsr.integrate(ts, mp, method='odeint')

    lsr_data = lsr.getOrbit()
    lsrR = RO * lsr_data[:,0]
    lsrphi = lsr_data[:,5]

    lsrX = lsrR * np.cos(lsrphi)
    lsrY = lsrR * np.sin(lsrphi)
    lsrZ = RO * lsr_data[:,3]

    R, vR, vT, z, vz, phi = 1., 0., 1., 0., 0., 0.25*np.pi
    rot_lsr_coords = [R, vR, vT, z, vz, phi]
    rot_lsr = Orbit(vxvv=rot_lsr_coords, solarmotion='schoenrich', vo=220, ro=8)
    rot_lsr.integrate(ts, mp, method='odeint')

    rot_lsr_data = rot_lsr.getOrbit()

    # putting into corotating cartesian system centred on LSR
    XYZUVW_rot = galpy_coords_to_xyzuvw(rot_lsr_data, ts)
    plt.clf()
    plt.plot(XYZUVW_rot[:,0], XYZUVW_rot[:,1])
    plt.savefig("temp_plots/rotXY.png")


    orbit_time = (perimeter / velocity).to("Myr")
    ts = np.linspace(0., 10*orbit_time.value, 1000) / BOVY_TIME_CONVERSION
    ra, dec, dist, mu_ra, mu_dec, vlos = 0., 0., 0., 0., 0., 0.
    solar_coords = [ra, dec, dist, mu_ra, mu_dec, vlos]
    sun = Orbit(vxvv=solar_coords, radec=True,
                solarmotion='schoenrich')  # should just be the sun's orbit
    sun.integrate(ts, mp, method='odeint')

    # get the orbit [R, vR, vT, z, vz, phi] (pos scaled by ro, vel scaled by vo)
    sun_data = sun.getOrbit()
    XYZUVW_sun = galpy_coords_to_xyzuvw(sun_data, ts)
    plt.clf()
    plt.plot(XYZUVW_sun[:,0], XYZUVW_sun[:,1])
    plt.savefig("temp_plots/sunXY.png")
    plt.clf()
    plt.plot(XYZUVW_sun[:,0], XYZUVW_sun[:,2])
    plt.savefig("temp_plots/sunXZ.png")

    #observed_to_xyzuvw(0, np.linspace(0,orbit_time.value, 100))
    galpy_ts = np.linspace(0, orbit_time.value, 100) / BOVY_TIME_CONVERSION
