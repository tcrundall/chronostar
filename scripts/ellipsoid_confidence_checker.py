#! /usr/bin/env python
"""
A little script to confirm the fraction of stars that fall within
a standard deviation from the mean.

Conclude that 68% of stars initialised as a spherical distribution with
std 10pc in radius, fall within 1.862 * 10pc of the origin.
"""
from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys
sys.path.insert(0, '..')

import chronostar.synthesiser as syn


def gaus(x, mu=0, sig=1):
    return 1./np.sqrt(2 * np.pi * sig**2) * np.exp(-(x-mu)**2 / (2*sig**2))


def area_ug(sigmas=1, mu=0, sig=1):
    npoints = 100
    limit = sigmas * sig
    xs = np.linspace(-limit, limit, npoints)
    dx = abs(xs[1] - xs[0])

    ys = gaus(xs, mu, sig)
    area = np.sum(ys) * dx
    return area


def get_distance(xyz, origin=(0,0,0)):
    dist = np.sqrt(np.sum((xyz - origin)**2))
    return dist


def get_distances(xyzs, origin=(0,0,0)):
    nstars = xyzs.shape[0]
    distances = np.zeros(nstars)
    for i, xyz in enumerate(xyzs):
        distances[i] = get_distance(xyz, origin)
    return distances

nstars = 10000
rd = 10.0 # radius
group_pars = np.array([0.,0.,0.,0,0,0,rd,rd,rd,2,0,0,0,0,nstars], dtype=float)

xyzuvw_now = syn.synth_group(group_pars)
distances = get_distances(xyzuvw_now[:,:3])

#conf_rad = 1.53 # # sigmas within which 68% of stars will be
conf_rad = 1.862

frac_with = np.sum(distances < conf_rad*rd) / nstars
print(frac_with)

