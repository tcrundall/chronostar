from galpy.potential import PowerSphericalPotentialwCutoff, MiyamotoNagaiPotential, NFWPotential, verticalfreq, MWPotential2014
from galpy.util import bovy_conversion
import numpy as np

import sys
sys.path.insert(0, '..')
from chronostar.traceorbit import trace_cartesian_orbit


def convert_myr2bovytime(times):
    """
    Convert times provided in Myr into times in bovy internal units.

    Galpy parametrises time based on the natural initialising values
    (r_0 and v_0) such that after 1 unit of time, a particle in a
    circular orbit at r_0, with circular velocity of v_0 will travel
    1 radian, azimuthally.

    Paramters
    ---------
    times : [ntimes] float array
        Times in Myr

    Return
    ------
    bovy_times : [ntimes] float array
        Times in bovy internal units
    """
    bovy_times = times*1e-3 / bovy_conversion.time_in_Gyr(220., 8.)
    return bovy_times

def convert_bovytime2myr(bovy_times):
    myr_times = bovy_times * bovy_conversion.time_in_Gyr(220., 8.) / 1e-3
    return myr_times


scale_height_factor = [0.5, 1.0, 2.0]
default_vper = 1./verticalfreq(MWPotential2014, 1.0)

print(convert_myr2bovytime(1))

import matplotlib.pyplot as plt
plt.clf()

colors = ['blue', 'purple', 'red']
for shf, color in zip(scale_height_factor, colors):
    bp= PowerSphericalPotentialwCutoff(alpha=1.8,rc=1.9/8.,normalize=0.05)
    mp= MiyamotoNagaiPotential(a=3./8.,b=shf*0.28/8.,
                               normalize=.6)
    nfwp= NFWPotential(a=16/8.,normalize=.35)

    my_mwpotential2014 = [bp,mp,nfwp]

    vfreq = verticalfreq(my_mwpotential2014, 1.0)
    vper = 1./vfreq

    if shf == 1.0:
        assert verticalfreq(my_mwpotential2014, 1.0) ==\
               verticalfreq(MWPotential2014, 1.0)

    print('_____ Scale height factor: {} _____'.format(shf))
    print('Vertical freq:                     {:6.1f}'.format(vfreq))
    print('Vertical period (perc of default): {:6.1f} %'.format(vper/default_vper * 100))
    print('Scaled vertical period:            {:6.1f} Myr'.format(convert_bovytime2myr(vper*2*np.pi)))
    print('')

    vertical_period = convert_bovytime2myr(vper*2*np.pi)

    init_xyzuvws = np.array([
        [0.,0.,25.,0.,0.,0.],
        [0.,0.,50.,0.,0.,0.],
        [0.,0.,100.,0.,0.,0.],
        [0.,0.,150.,0.,0.,0.],
    ])

    maxtime = 160.
    times = np.linspace(0,maxtime, 200)
    for init_xyzuvw, ls in zip(init_xyzuvws, ['-', '-.', '--', ':']):
        cart_orbit = trace_cartesian_orbit(init_xyzuvw, times=times,
                                           single_age=False,
                                           potential=my_mwpotential2014)
        plt.plot(times, cart_orbit[:,2], #label='shf={:.1f}, z0={}'.format(shf, init_xyzuvw[2]),
                 color=color, linestyle=ls)
    plt.plot(vertical_period, 0, color=color, marker='.', markersize=20)

plt.legend(loc='best')
plt.savefig('../plots/potential_oscillation.png')
