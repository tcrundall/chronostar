"""Let's:

1) [just for fun] Trace a star back in time to get us an initial XYZUVW at known delta t.
2) Trace the same star forward in time the same known delta t, and add noise.
3) Trace this back with covariance matrix at many times in order to be input for a group fitter.
"""
from __future__ import division, print_function
import numpy as np
from astropy import units as u 
from astropy.coordinates import SkyCoord
import chronostar.traceback as tb

#Start with the coordinates of beta Pic.
star_radecpipmrv = [86.82, -51.067, 51.44, 4.65, 83.1, 20]
age = 20. # 20.0

#Trace the star back.
tracer = tb.TraceBack(params=star_radecpipmrv)
xyzuvw_back = tracer.traceback(np.array([0,age]))

#Now trace it forward!
sky_coord_then = tb.xyzuvw_to_skycoord(xyzuvw_back[0][-1], solarmotion='schoenrich', reverse_x_sign=True)
xyzuvw_now = tb.trace_forward(sky_coord_then, age)
sky_coord_now = tb.xyzuvw_to_skycoord(xyzuvw_now, solarmotion='schoenrich', reverse_x_sign=True)

#Find the difference...
print(sky_coord_now - np.array(star_radecpipmrv))