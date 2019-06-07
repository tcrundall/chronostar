import numpy as np
import sys
import time
sys.path.insert(0, '..')

from chronostar.traceorbit import trace_cartesian_orbit

integ_methods = ['odeint', 'dopr54_c']


xyzuvw_start = [0.,0.,25.,0.,0.,0.]
max_time = 1000
orbit_times = np.linspace(0,max_time,100)

niters = 100

print('Integrating up to {} Myr'.format(max_time))
print('Iterating {} times'.format(niters))

for method in integ_methods:
    print('Using {}'.format(method))
    duration_times = []
    for i in range(niters):
        start = time.clock()

        trace_cartesian_orbit(xyzuvw_start, orbit_times, single_age=False,
                              method=method)

        end = time.clock()
        duration_times.append(end-start)
    print('Average time taken: {:.1f} ms'.format(1000*np.mean(duration_times)))
    print('Best time:          {:.1f} ms'.format(1000*np.min(duration_times)))
    print('Worst time:         {:.1f} ms'.format(1000*np.max(duration_times)))


