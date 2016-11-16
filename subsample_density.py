"""Create a subsample based on a given time in the past for all stars.

Then for this subsample, find the sum of the density functions for all stars.

We will use:
numpy.random.multivariate_normal

Then simply:

(x-y)^T C^-1 (x-y) for a log(probability)

This needs an npoints x 6 x 6 matrix
"""

from __future__ import division, print_function
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
import chronostar
plt.ion()

t_ixs = np.arange(0,11)  #Start with a fixed time index (5 is 15.0 Myr)
nsamp = 16 #Number of samples per star
x_grid = np.linspace( -200, 200, 100)
y_grid = np.linspace(-200, 200, 100)
smooth_pc = 5

infile = '/Users/mireland/Google Drive/chronostar_catalogs/TGAS_traceback_165Myr_small.fits'

#--------

#Read in the parameters
star_params = chronostar.fit_group.read_stars(infile)

for t_ix in t_ixs:
    #Extract our fixed times.
    xyzuvw = star_params['xyzuvw'][:,t_ix,:]
    xyzuvw_cov = star_params['xyzuvw_cov'][:,t_ix,:]
    xyzuvw_icov = star_params['xyzuvw_icov'][:,t_ix,:]

    ns = len(xyzuvw) #Number of stars

    #Create our random sample.
    print("Creating Sample...")
    sample_xyzuvw = np.zeros( (ns, nsamp, 6) )
    for i in range(ns):
        sample_xyzuvw[i] = np.random.multivariate_normal(xyzuvw[i], xyzuvw_cov[i],size=nsamp)
    sample_xyzuvw = sample_xyzuvw.reshape( (ns*nsamp, 6) )

    print("Computing Density...")
    density = np.zeros( (len(x_grid), len(y_grid)), dtype=np.int16)
    tree_xy = cKDTree(sample_xyzuvw[:,0:2])
    for x_ix, x in enumerate(x_grid):
        for y_ix, y in enumerate(y_grid):
            density[y_ix, x_ix] = len(tree_xy.query_ball_point([x,y], smooth_pc))

    plt.clf()     
    plt.imshow(density, extent=[np.min(x_grid), np.max(x_grid), np.min(y_grid), np.max(y_grid)])
    plt.title('{0:5.1f} Myr'.format(star_params['times'][t_ix]))
    plt.xlabel('X (pc)')
    plt.ylabel('Y (pc)')
    plt.pause(0.001)
    plt.savefig('imgs/{0:02d}'.format(t_ix))