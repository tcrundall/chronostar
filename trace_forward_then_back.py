"""Let's:

1) [just for fun] Trace a star back in time to get us an initial XYZUVW at known delta t.
2) Trace the same star forward in time the same known delta t, and add noise.
3) Trace this back with covariance matrix at many times in order to be input for a group fitter.

Problems:
a: Even lsr_orbit doesn't work. 
lsr_orbit = tb.get_lsr_orbit(np.linspace(0,1,10))
lsr_orbit.V(0) is correct, but lsr_orbit.U(0) and lsr_orbit.W(0) are out by 30m/s.

b: There is up to a 5 pc difference between initial and final positions.

"""
from __future__ import division, print_function
import numpy as np
from astropy import units as u 
from astropy.coordinates import SkyCoord
import chronostar.traceback as tb

#Start with the coordinates of beta Pic.
star_radecpipmrv = [86.82, -51.067, 51.44, 4.65, 83.1, 20]

#To test a very nearby star, uncomment the following line...
#star_radecpipmrv = [0,0, 1e12, 0,0,0]

age = 20.
#age = 1e-3

#First, the most manual traceback we can do.
xyzuvw_back = tb.integrate_xyzuvw(star_radecpipmrv,np.array([0,age]))

#Trace the star back, giving an xyzuvw relative to the local standard of rest.
if (False):
    stars = tb.stars_table(star_radecpipmrv)
    xyzuvw_back = tb.traceback(stars, np.array([0,age]))[0]

#Now trace it forward!
xyzuvw_now = tb.trace_forward(xyzuvw_back[-1], age, solarmotion=None)

print("XYZUVW difference")
print(xyzuvw_now - np.array(xyzuvw_back[0]))

sky_coord_now = tb.xyzuvw_to_skycoord(xyzuvw_now, solarmotion='schoenrich', reverse_x_sign=True)

print("Sky Coord difference")
print(sky_coord_now - np.array(star_radecpipmrv))

if (False):
    #Original trace forward attempt with sky coordinates.
    sky_coord_orig = tb.xyzuvw_to_skycoord(xyzuvw_back[0], solarmotion='schoenrich', reverse_x_sign=True)
    sky_coord_then = tb.xyzuvw_to_skycoord(xyzuvw_back[-1], solarmotion='schoenrich', reverse_x_sign=True)

    xyzuvw_now = tb.trace_forward_sky(sky_coord_then, age)
    sky_coord_now = tb.xyzuvw_to_skycoord(xyzuvw_now, solarmotion='schoenrich', reverse_x_sign=True)

    #Find the difference...
    print(sky_coord_now - np.array(star_radecpipmrv))