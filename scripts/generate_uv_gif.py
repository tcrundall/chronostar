#! /usr/bin/env python

import sys
sys.path.insert(0,'..')

import numpy as np
import pickle
import pdb
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

def cart_to_polar(x,y):
    theta = np.arctan(x/y) * 180.0 / np.pi + 90
    # convoluted way to handle tan degeneracy without an if condition
    theta += 180*(-0.5*np.sign(y) + 0.5)
    return theta

def star_in_range(long_min, long_max, xyzuvw):
    """Given a galactic longtitude range, see if star falls in range
    """
    x = xyzuvw[0]
    y = xyzuvw[1]
    theta = cart_to_polar(x, y)
    return long_min < theta and theta < long_max

def get_slice(long_min, long_max, xyzuvws):
    xs = xyzuvws[:,0,0]
    ys = xyzuvws[:,0,1]
    ixs = np.where(
        np.logical_and(
            cart_to_polar(xs, ys) > long_min,
            cart_to_polar(xs, ys) < long_max
        )
    )
    return ixs

def test_cart_to_polar():
    xys = [
        [-1, 1e-10],
        [-1, 1],
        [1e-10, 1],
        [1, 1],
        [1, 1e-10],
        [1, -1],
        [1e-10, -1],
        [-1, -1],
    ]

    thetas = [0, 45, 90, 135, 180, 225, 270, 315]
    
    for xy, theta in zip(xys, thetas):
        assert np.isclose(theta, cart_to_polar(xy[0], xy[1]))
        #print(theta, cart_to_polar(xy[0], xy[1]))
    
#test_cart_to_polar()

tb_file = '../data/tb_rave_active_star_candidates_with_TGAS_kinematics.pkl'
with open(tb_file, 'r') as fp:
    t, _, xyzuvw, xyzuvw_cov = pickle.load(fp)

#nbins = 200
#minimum_long = 0.0
#maximum_long = 360.0

bins = 36
statistic, bin_edges, binnumber = binned_statistic(
    cart_to_polar(xyzuvw[:,0,0], xyzuvw[:,0,1]),
    [xyzuvw[:,0,3], xyzuvw[:,0,4]], bins=bins
)
#pdb.set_trace()

nbins = bin_edges.shape[0] - 1
for i in range(nbins):
    plt.clf()
    us = xyzuvw[np.where(binnumber == (i+1)),0,3]
    vs = xyzuvw[np.where(binnumber == (i+1)),0,4]
    plt.plot(us, vs, '.')
    plt.title(r"Galactic Longtitude: ${}^\circ - {}^\circ$".\
        format(
            int(np.around(bin_edges[i],   -1)),
            int(np.around(bin_edges[i+1], -1))
        )
    )
    #pdb.set_trace()
    plt.xlabel("U [km/s]")
    plt.ylabel("V [km/s]")
    plt.ylim(-100, 50)
    plt.xlim(-100, 100)
    plt.savefig("uv_gif/{}.png".format(i))

    plt.clf()
    xs = xyzuvw[np.where(binnumber == (i+1)),0,0]
    ys = xyzuvw[np.where(binnumber == (i+1)),0,1]
    plt.plot(xs, ys, '.')
    plt.title(r"Galactic Longtitude: ${}^\circ - {}^\circ$".\
        format(
            int(np.around(bin_edges[i],   -1)),
            int(np.around(bin_edges[i+1], -1))
        )
    )
    #pdb.set_trace()
    plt.xlabel("X [pc]")
    plt.ylabel("Y [pc]")
    plt.ylim(-200, 200)
    plt.xlim(-200, 200)
    plt.savefig("xy_gif/{}.png".format(i))


ixs = get_slice(0, 90, xyzuvw)

nstars = xyzuvw.shape[0]

# Bin stars into galactic longtitude

