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
    plt.close('all')

    f, (ax1, ax2) = plt.subplots(1,2)

    us = xyzuvw[np.where(binnumber == (i+1)),0,3]
    vs = xyzuvw[np.where(binnumber == (i+1)),0,4]
    ax1.set(aspect='equal')
    ax1.plot(us, vs, '.')
    ax1.set_xlabel("U [km/s]")
    ax1.set_ylabel("V [km/s]")
    vrange = 150
    ax1.set_ylim(-vrange, vrange)
    ax1.set_xlim(vrange, -vrange)  # note inversion

    xs = xyzuvw[np.where(binnumber == (i+1)),0,0]
    ys = xyzuvw[np.where(binnumber == (i+1)),0,1]
    ax2.set(aspect='equal')
    ax2.plot(xs, ys, '.')
    pos_range = 300
    ax2.set_xlabel("X [pc]")
    ax2.set_ylabel("Y [pc]")
    ax2.set_ylim(-pos_range, pos_range)
    ax2.set_xlim(pos_range, -pos_range) # note inversion

    f.suptitle(r"Galactic Longtitude: ${}^\circ - {}^\circ$".\
        format(
            int(np.around(bin_edges[i],   -1)),
            int(np.around(bin_edges[i+1], -1))
        )
    )
    f.tight_layout(pad=2.0)
    f.savefig("combined_gif/{}.png".format(i))
