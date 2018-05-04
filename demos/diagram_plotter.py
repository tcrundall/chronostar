"""
Simple script producing a diagram that demos how fits will be initialised
"""
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0,'..')

import chronostar.synthesiser as syn
import chronostar.errorellipse as ee
import chronostar.expectmax as em

group_pars = np.array([
    [0, 0, 0, 50, -10, 0, 10, 25, 0, 100],
    [0, 0, 0, -20, 35, 0, 10, 15, 0, 60],
    [0, 0, 0, -25, -50, 0, 10, 25, 0, 40],
])
ngroups = group_pars.shape[0]
nstars = np.sum(group_pars[:,-1])

groups = ngroups*[None]

xyzuvw = np.zeros((0,6))
dividers = [0]

for i in range(ngroups):
    groups[i] = syn.Group(group_pars[i])
    xyzuvw = np.vstack((xyzuvw, syn.synthesise_xyzuvw(
        group_pars[i]
    )))
    dividers.append(dividers[-1] + group_pars[i,-1])

plt.clf()
for i in range(ngroups):
    plt.plot(xyzuvw[dividers[i]:dividers[i+1],3],
             xyzuvw[dividers[i]:dividers[i+1],4], '.',
             label="group {}".format(i))
u_mn, v_mn = np.mean(xyzuvw, axis=0)[3:5]
plt.plot(u_mn, v_mn, '+', label="UV mean")

init_comps = em.get_initial_group_pars(ngroups, xyzuvw, refl=True)
comp_groups = ngroups * [None]

for i in range(ngroups):
    comp_groups[i] = syn.Group(init_comps[i], internal=True, starcount=False)
    ee.plotCovEllipse(comp_groups[i].generateCovMatrix()[3:5, 3:5],
                      comp_groups[i].mean[3:5], with_line=True)
plt.legend(loc='best')
plt.xlabel('U [km/s]')
plt.ylabel('V [km/s]')
plt.savefig("demo_diagram.png")

