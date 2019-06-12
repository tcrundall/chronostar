import numpy as np



from galpy.actionAngle import actionAngleTorus
from galpy.potential import MWPotential2014

sys.path.insert(0, '..')

import chronostar.traceorbit as torb

xyzuvw_start = [0.,0.,25.,0.,0.,0.]
print('xyzuvw start: {}'.format(xyzuvw_start))

galpy_coords = torb.convert_cart2galpycoords(xyzuvw_start)
print('galpy start: {}'.format(galpy_coords))



