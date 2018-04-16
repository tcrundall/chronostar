from galpy.orbit import Orbit
from galpy.potential import MWPotential2014 as mp
import matplotlib.pyplot as plt
import numpy as np

# GALPY STORES ORBITAL INFO IN GALACTOCENTRIC CYLINDRICAL COORDINATES
# [R, vR, vT, z, vZ, phi]

def galpy_coords_to_xyzuvw(data, ro=8., vo=220.):
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
    ro : float
        a conversion factor that takes units from galpy units to
        physical units. If left as default, output will be in kpc
    vo : float
        a conversion factor that takes units form galpy units to
        physical units. If left as default, output will be in km/s
        This is also the circular velocity of a circular orbit with X,Y
        equal to that of the sun.
    """
    return None

#def demo_lsr_and_sun_calc():
if __name__ == '__main__':
    max_age = 1 #Gyr
    ntimes = 100
    ts= np.linspace(0,max_age,ntimes)

    # INITIALISING SUN COORDINATES AND ORBIT
    #deg, deg, kpc,  mas/yr, mas/yr, km/s
    ra,   dec, dist, mu_ra,  mu_dec, vlos = 0., 0., 0., 0., 0., 0.
    solar_coords = [ra, dec, dist, mu_ra, mu_dec, vlos]
    sun = Orbit(vxvv=solar_coords, radec=True, solarmotion='schoenrich') # should just be the sun's orbit
    sun.integrate(ts,mp,method='odeint')

    # get the orbit (not sure what coords...)
    sun_data = sun.getOrbit()

    # plots the sun's motion with respect to LSR(?)
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

    # plot the XY of the sun's motion using galpy's plot function
    plt.clf()
    sun.plot(d1='x', d2='y')
    plt.savefig('temp_plots/galpy_sunXY.png')

    sun.plot(d1='x', d2='z')
    plt.savefig('temp_plots/galpy_sunXZ.png')

    #                                                        kpc, km/s
    # INITIALISING THE LSR (at XYZUVW (w.r.t sun) of [0,0,-0.025,0,220,0]
    R, vR, vT, z, vz, phi = 1., 0., 1., 0., 0./220, 0.
    LSR_coords = [R, vR, vT, z, vz, phi]
    lsr = Orbit(vxvv=LSR_coords, solarmotion='schoenrich', vo=220, ro=8)
    lsr.integrate(ts, mp, method='odeint')

    plt.clf()
    # SHOULD PLOT A PERFECT CIRCLE
    lsr.plot(d1='x', d2='y')
    plt.savefig('temp_plots/galpy_lsrXY.png')

    plt.clf()
    lsr.plot(d1='x', d2='z')
    plt.savefig('temp_plots/galpy_lsrXZ.png')

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
    #plt.show()


#if __name__ == '__main__':
def lsr_nonsense():
    max_age = 4. #Gyr
    ntimes = 100
    ts= np.linspace(0,max_age,ntimes)

    # demo a star (with vT=220, vR=0, vZ=0, z=0, phi=0.1 pi) staying
    # fixed in our coordinate frame
    R, vR, vT, z, vz, phi = 1., 0., 1., 0., 0./220, 0.
    LSR_coords = [R, vR, vT, z, vz, phi]
    lsr = Orbit(vxvv=LSR_coords, solarmotion='schoenrich', vo=220, ro=8)
    lsr.integrate(ts, mp, method='odeint')

    lsr_data = lsr.getOrbit()
    lsrR = 8 * lsr_data[:,0]
    lsrphi = lsr_data[:,5]

    lsrX = lsrR * np.cos(lsrphi)
    lsrY = lsrR * np.sin(lsrphi)
    lsrZ = 8 * lsr_data[:,3]

    R, vR, vT, z, vz, phi = 1., 0., 1., 0., 0./220, 0.25*np.pi
    rot_lsr_coords = [R, vR, vT, z, vz, phi]
    rot_lsr = Orbit(vxvv=rot_lsr_coords, solarmotion='schoenrich', vo=220, ro=8)
    rot_lsr.integrate(ts, mp, method='odeint')

    rot_lsr_data = rot_lsr.getOrbit()
    rot_lsrR = 8 * rot_lsr_data[:,0]
    rot_lsrphi = rot_lsr_data[:,5]

    rot_lsrX = rot_lsrR * np.cos(rot_lsrphi)
    rot_lsrY = rot_lsrR * np.sin(rot_lsrphi)
    rot_lsrZ = 8 * rot_lsr_data[:,3]

    plt.clf()
    plt.plot(lsrX, lsrY + 1)
    plt.plot(rot_lsrX, rot_lsrY)
    plt.savefig('temp_plots/rotXY.png')
    plt.clf()
    plt.plot(lsrX, lsrZ)
    plt.plot(rot_lsrX, rot_lsrZ)
    plt.savefig('temp_plots/rotXZ.png')

    # putting into corotating cartesian system centred on LSR
    rel_data = rot_lsr_data - lsr_data


