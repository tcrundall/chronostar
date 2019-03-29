from __future__ import print_function, division, unicode_literals

import sys
sys.path.insert(0, '..')

from chronostar import traceorbit
from galpy.potential import MiyamotoNagaiPotential, MWPotential2014 #, DehnenBarPotential, KeplerPotential

miya_pot = MiyamotoNagaiPotential(a=0.5,b=0.0375,amp=1.,normalize=1.) # Params from the example webpage. No idea if that's good or not.


if __name__ == '__main__':
    def fungetter(potential):
        def f_(xyzuvw_start, times):
            return traceorbit.trace_cartesian_orbit(xyzuvw_start, times,
                                                    potential=potential)
        return f_

    tweaked_trace_orbit = fungetter(miya_pot)
