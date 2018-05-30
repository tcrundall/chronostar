from __future__ import division, print_function
"""
Simple script that plots the XY and UV of stars from Federrath 2015
"""
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '..')
import chronostar.groupfitter as gf

COLORS = ['xkcd:orange', 'xkcd:cyan',
          'xkcd:sun yellow', 'xkcd:shit', 'xkcd:bright pink']


def plot_pair(dim1, dim2, xyzuvw, masks=None):
    if not masks:
        masks = [np.where(xyzuvw)]
    labels =['X [pc]', 'Y [pc]', 'Z [pc]',
             'U [km/s]', 'V [km/s]', 'W [km/s]']
    for i, mask in enumerate(masks):
        plt.plot(xyzuvw[mask,dim1], xyzuvw[mask,dim2], '.', color=COLORS[i])
    plt.xlabel(labels[dim1])
    plt.ylabel(labels[dim2])


if __name__ == '__main__':
    fed_stars = np.load("../data/sink_init_xyzuvw.npy")

    # masks determined by eye
    mask_a = np.where(fed_stars[:,0] < -0.6)
    mask_b = np.where((fed_stars[:,0] > -0.6) & (fed_stars[:,0] < 0.))
    mask_c = np.where((fed_stars[:,0] > 0.) & (fed_stars[:,0] < 0.6))
    mask_d = np.where(fed_stars[:,0] > 0.6)

    masks = [mask_a, mask_b, mask_c, mask_d]

    plt.clf()
    plot_pair(0,1,fed_stars, masks)
    plt.savefig("XY.pdf")
    plt.clf()
    plot_pair(0,2,fed_stars, masks)
    plt.savefig("XZ.pdf")
    plt.clf()
    plot_pair(1,2,fed_stars, masks)
    plt.savefig("YZ.pdf")
    plt.clf()
    plot_pair(3,4,fed_stars, masks)
    plt.savefig("UV.pdf")
    plt.clf()
    plot_pair(3,5,fed_stars, masks)
    plt.savefig("UW.pdf")
    plt.clf()
    plot_pair(4,5,fed_stars, masks)
    plt.savefig("VW.pdf")
    plt.clf()
    plot_pair(0,3,fed_stars, masks)
    plt.savefig("XU.pdf")
    plt.clf()
    plot_pair(1,4,fed_stars, masks)
    plt.savefig("YV.pdf")
    plt.clf()
    plot_pair(2,5,fed_stars, masks)
    plt.savefig("ZW.pdf")



