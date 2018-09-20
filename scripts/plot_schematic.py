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
import chronostar.fitplotter as fp

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

plt.clf()
ax = plt.subplot()
fp.plotPane(dim1='z', dim2='w', ax=ax, groups=origin, star_pars=star_pars,
            group_then=True, group_now=True, group_orbit=True, annotate=True)
plt.savefig(pdir + "module-ZW-schematic.pdf", bbox_inches='tight')


