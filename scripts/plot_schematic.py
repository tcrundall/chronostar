from __future__ import print_function, division
"""Generate a diagram detailing model fitting approach"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
import chronostar.synthesiser as syn
import chronostar.traceorbit as torb
import chronostar.measurer as ms
import chronostar.converter as cv
import chronostar.hexplotter as hp
import chronostar.errorellipse as ee
import chronostar.transform as tf


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )

pdir = "../figures/paper1/"
ERROR = 1.0

origin_pars = np.array([500., 0., -50., 20., 0., 0., 10., 0.5, 25., 10])

xyzuvw_init, origin = syn.synthesiseXYZUVW(origin_pars, sphere=True,
                                           return_group=True,
                                           internal=False)
xyzuvw_now_perf = torb.traceManyOrbitXYZUVW(xyzuvw_init,
                                            times=origin.age,
                                            single_age=True)

astr_table = ms.measureXYZUVW(xyzuvw_now_perf, ERROR)
star_pars = cv.convertMeasurementsToCartesian(astr_table)

mns = star_pars['xyzuvw']
cov_then = origin.generateSphericalCovMatrix()
mean_then = origin.mean
cov_now = tf.transform_cov(cov_then, torb.traceOrbitXYZUVW,
                           mean_then, args=[origin.age])
mean_now = torb.traceOrbitXYZUVW(mean_then, origin.age, single_age=True)
mean_orbit = torb.traceOrbitXYZUVW(mean_then, np.linspace(0,origin.age,50),
                                   single_age=False)

plt.clf()
dim1 = 2
dim2 = 5
labels = 'XYZUVW'
units = 3*['pc'] + 3*['km/s']
ax = plt.subplot()
ax.set_xlabel("{} [{}]".format(labels[dim1], units[dim1]))
ax.set_ylabel("{} [{}]".format(labels[dim2], units[dim2]))
ax.plot(mns[:,dim1], mns[:,dim2], '.')
orbit_traj =\
    ax.plot(mean_orbit[:,dim1], mean_orbit[:,dim2], ls='-', alpha=0.1,
            color='xkcd:grey')
ax.plot(mean_then[dim1], mean_then[dim2], marker='x',
        color='xkcd:grey')
ee.plotCovEllipse(cov_then[np.ix_([dim1,dim2],[dim1,dim2])],
                  mean_then[np.ix_([dim1,dim2])],
                  with_line=True,
                  ax=ax, color="xkcd:grey", alpha=0.3, ls='--')
ax.plot(mean_now[dim1], mean_now[dim2], marker='x',
        color='xkcd:blue')
ee.plotCovEllipse(cov_now[np.ix_([dim1,dim2],[dim1,dim2])],
                  mean_now[np.ix_([dim1,dim2])],
                  with_line=True,
                  ax=ax, color="xkcd:blue", alpha=0.1, ls='--')

mid_orbit_ix = int(0.5*len(mean_orbit))
# ax.text(mean_orbit[mid_orbit_ix, dim1], mean_orbit[mid_orbit_ix,dim2],
#          "Orbital trajectory")

ax.annotate("Orbital trajectory", (mean_orbit[mid_orbit_ix, dim1],
                                   mean_orbit[mid_orbit_ix,dim2]), )
add_arrow(orbit_traj[0], position=-40,
          color='xkcd:grey')
add_arrow(orbit_traj[0], position=-20,
          color='xkcd:grey')
add_arrow(orbit_traj[0], position=0,
          color='xkcd:grey')
ax.annotate(r'$\mathbf{\mu}_0, \mathbf{\Sigma}_0$', (mean_then[dim1],
                                   mean_then[dim2]))
ax.annotate(r'$\mathbf{\mu}_c, \mathbf{\Sigma}_c$', (mean_now[dim1],
                                   mean_now[dim2]),
            color='xkcd:blue')
# ax.annotate(mean_now[dim1] + 1, mean_now[dim2] + 1, r'$\mu_c, \Sigma_c$',
#         color='xkcd:blue')

plt.savefig(pdir + "ZW-schematic.pdf", bbox_inches='tight')


