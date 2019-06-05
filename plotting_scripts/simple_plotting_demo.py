import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from chronostar.component import SphereComponent
from chronostar.traceorbit import trace_cartesian_orbit

import numpy as np


mean_now = np.array([0., 0., 30., 5., 5., 5.])
init_dx = 5.
init_dv = 1.
age = 100.

mean_then = trace_cartesian_orbit(mean_now, times=-age)

pars1 = np.hstack((mean_then, init_dx, init_dv, age))
comp1 = SphereComponent(pars1)
print(comp1.get_pars())

labels = 'XYZUVW'
units = 3*['pc'] + 3*['km/s']

for dim1, dim2 in [(0,3), (1,4), (2,5)]:
    plt.clf()
    comp1.plot(dim1=dim1, dim2=dim2, comp_now=True, comp_then=True,
               comp_orbit=True)
    plt.xlabel('{} [{}]'.format(labels[dim1], units[dim1]))
    plt.ylabel('{} [{}]'.format(labels[dim2], units[dim2]))
    plt.savefig('../plots/simple_plot_{}{}.pdf'.format(labels[dim1],
                                                       labels[dim2]))