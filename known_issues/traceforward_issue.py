'''
Noticed issue with calculating a component's current
day mean. Behaviour of trace_cartesian_orbit varies based
on whether a single age is provided vs a range of times
'''

import numpy as np
import sys

sys.path.insert(0, '..')
from chronostar.component import SphereComponent
from chronostar.traceorbit import trace_cartesian_orbit

bugged_comp = SphereComponent.load_raw_components('bugged_component.npy')[0]

y_pos_now = bugged_comp.get_mean_now()[1]
print('Mean then: {}'.format(bugged_comp.get_mean()))
print('Internally calculated mean_now: {}'.format(bugged_comp.get_mean_now()))
mean_then = bugged_comp.get_mean()
age = bugged_comp.get_age()
y_pos_linear = mean_then[1] + mean_then[4]*age
print('Linear Y motion: {}'.format(y_pos_linear))

print('Difference of {}'.format(y_pos_now - y_pos_linear))

ts = np.linspace(0,bugged_comp.get_age(),50)
for method in ['odeint', 'symplec4_c', 'rk4_c', 'dopr54_c', 'rk6_c',]:
    manual_single_age_mean_now = trace_cartesian_orbit(bugged_comp.get_mean(),
                                                       times=bugged_comp.get_age(),
                                                       method=method)

    manual_multi_age_mean_now = trace_cartesian_orbit(bugged_comp.get_mean(),
                                                      times=ts,
                                                      method=method,
                                                      single_age=False)
    print(method)
    print('Manual Y now (single age): {}'.format(manual_single_age_mean_now[1]))
    print('Manual Y now (multi age):  {}'.format(manual_multi_age_mean_now[-1,1]))
