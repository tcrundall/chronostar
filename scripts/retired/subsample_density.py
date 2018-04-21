#!/usr/bin/env python
"""Create a subsample based on a given time in the past for all stars.

Then for this subsample, find the sum of the density functions for all stars.

We will use:
numpy.random.multivariate_normal

Then simply:

(x-y)^T C^-1 (x-y) for a log(probability)

This needs an npoints x 6 x 6 matrix

ffmpeg -i d%03d.png -r 8 -y traceback.mp4


Finding a new group...
diff = xyzuvw - np.tile(xyzuvw_traceback_groups[4][-1],65310).reshape(65310,6)
dist = np.array([np.sum(d**2/np.array([30,30,140,4,4,10])**2) for d in diff])
poss_memb = np.where(dist<100)[0]
wcore = (star_params['stars']['ra_adopt'][poss_memb]>175) * (star_params['stars']['ra_adopt'][poss_memb]<205)*(star_params['stars']['dec_adopt'][poss_memb]>18)*(star_params['stars']['dec_adopt'][poss_memb]<35)
core_memb = poss_memb[np.where(wcore)[0]]
noncore_memb = poss_memb[np.where(1- wcore)[0]]
"""
from __future__ import division, print_function

from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
import chronostar
import pdb

plt.ion()

#t_ixs = np.arange(0,11)  #Start with a fixed time index (5 is 15.0 Myr)
#t_ixs = np.arange(1,2)  #Start with a fixed time index (5 is 15.0 Myr)

target_times = np.arange(101)
target_times = np.array([0,50])

nsamp = 32 #Number of samples per star
n_neighbors = 256
axis_lim = 2000
x_grid = np.linspace( -axis_lim, axis_lim, 400)
y_grid = np.linspace(-axis_lim, axis_lim, 400)
smooth_pc = 15
plane_thickness=280 #2 Scale heights

infile = '/Users/mireland/Google Drive/chronostar_catalogs/TGAS_traceback_165Myr_small.fits'

#From Malo (2013), with the X coordinates changed to match our sign convention.
xyzuvw_bpic = [-9.27,-5.96,-13.59,-10.94,-16.25,-9.27]
xyzuvw_tha = [-11.39,-21.21,-35.40,-9.88,-20.70,-0.9]
xyzuvw_abdmg = [2.37,1.48,-15.62,-7.12,-27.31,-13.81]
xyzuvw_col = [27.44,-31.32,-27.97,-12.24,-21.32,-5.58]
xyzuvw_car = [-15.55,-58.53,-22.95,-10.5,-22.36,-5.84]
xyzuvw_twa = [-12.49,-42.28,21.55,-9.87,-18.06,-4.52]
xyzuvw_arg = [-1.32,1.97,0.50,-21.78,-12.08,-4.52]
#Roser11, relative to sun. Again - we have to reverse the X axis
xyzuvw_hyades =[43.1, 0.7, -17.3, -41.1, -19.2, -1.4] 
#From the brightest 3 stars in Kraus
#xyzuvw_comaber = [ 10.127,  -8.276,  92.509,  -1.182,  -5.486,   0.282] 
xyzuvw_comaber = [  9.66 ,  -7.368,  85.717,  -1.887,  -5.712,   0.643]
#From the Simbad coordinates of Plieades:
#pleiades = chronostar.tracingback.TraceBack(target_times, params=[15*(3+47./60),24.1,1e3/136.2,19.7,-44.82,3.5]).traceback(target_times)[0]
#pleiades[0] - xyzuvw_sun
xyzuvw_pleiades = [ 121.633,   28.975,  -54.003,   -4.857,  -28.54 ,  -13.139]
#usco = chronostar.tracingback.TraceBack(target_times, params=[15*(16+12./60),-23.5,1e3/145.,-11,-23,-5]).traceback(target_times)[0]
xyzuvw_usco = [-134.927,  -20.315,   49.062,   -5.347,  -16.07 ,   -6.713]

#Shoenrich and galpy
xyzuvw_sun = np.array([0,0,25,11.1,12.24,7.25])
groups = ['bPic','THA','ABD','Hyades','ComaBer','Pleiades','USco','Sun']
xyzuvw_groups = np.array([xyzuvw_bpic, xyzuvw_tha, xyzuvw_abdmg, xyzuvw_hyades, xyzuvw_comaber, xyzuvw_pleiades, xyzuvw_usco])
for xyzuvw in xyzuvw_groups:
    xyzuvw += xyzuvw_sun
xyzuvw_groups = np.concatenate((xyzuvw_groups, [xyzuvw_sun]))

savefigs = False
compute_density = False

#---------------------------

#Trace back the groups
xyzuvw_traceback_groups = []
for xyzuvw in xyzuvw_groups:
    params = chronostar.retired.tracingback.xyzuvw_to_skycoord(xyzuvw, solarmotion='schoenrich', reverse_x_sign=True)
    xyzuvw_traceback_groups.append(
        chronostar.retired.tracingback.TraceBack(target_times, params=params).traceback(target_times)[0])

#Read in the parameters
star_params = chronostar.retired.fit_group.read_stars(infile)

#Some covariance matrices are bad...
for t_ix, target_time in enumerate(target_times):
    xyzuvw, xyzuvw_cov = chronostar.retired.fit_group.interp_cov(target_time, star_params)
    ns = len(xyzuvw) #Number of stars
    xyzuvw_cov += np.tile(np.diag([2,2,2,1,1,1]).flatten(), ns).reshape(ns, 6,6)
    xyzuvw_icov = np.linalg.inv(xyzuvw_cov)
    xyzuvw_icov_det = np.linalg.det(xyzuvw_icov)
    if False:
        #Extract our fixed times.
        xyzuvw = star_params['xyzuvw'][:,t_ix,:]
        xyzuvw_cov = star_params['xyzuvw_cov'][:,t_ix,:]
        #!!! WARNING: It would be better here to explicitly add a "typical" 1km/s and 2pc bubble.
        xyzuvw_icov = star_params['xyzuvw_icov'][:,t_ix,:]
        xyzuvw_icov_det = star_params['xyzuvw_icov_det'][:,t_ix]
    
    #!!! WARNING: The following line shouldn't be needed as no covariance matrix should have a negative determinant!
    xyzuvw_good = np.empty(ns, dtype=np.bool)
    for i in range(ns):
        xyzuvw_good[i] = np.linalg.eigvalsh(xyzuvw_icov[i])[0] > 0
 
    #Create our random sample.
    print("Creating Sample...")
    sample_xyzuvw = np.zeros( (ns, nsamp, 6) )
    for i in range(ns):
        if xyzuvw_good[i]:
            sample_xyzuvw[i] = np.random.multivariate_normal(xyzuvw[i], xyzuvw_cov[i],size=nsamp)
        else:
            for j in range(nsamp):
                sample_xyzuvw[i,j] = xyzuvw[i]
    sample_xyzuvw = sample_xyzuvw.reshape( (ns*nsamp, 6) )

    if compute_density:
        print("Computing Density at each sample point...")
        stars_tree = cKDTree(xyzuvw)
        sample_density = np.zeros( (ns, nsamp) )
        self_density = np.zeros( (ns, nsamp) )
        for i in range(ns):
            for j in range(nsamp):
                nearby = stars_tree.query(sample_xyzuvw[i*nsamp + j], n_neighbors) 
                for ix in nearby[1]:
                    if xyzuvw_good[ix] and i != ix:
                        diff = xyzuvw[ix] - sample_xyzuvw[i*nsamp + j]
                        sample_density[i,j] += np.exp(-0.5*np.dot(diff,np.dot(xyzuvw_icov[ix],diff)))*np.sqrt(xyzuvw_icov_det[ix])
                #if xyzuvw_good[i]:
                #    diff = xyzuvw[i] - sample_xyzuvw[i*nsamp + j]
                #    sample_density[i,j] -= np.exp(-0.5*np.dot(diff, np.dot(xyzuvw_icov[i], diff)))*np.sqrt(xyzuvw_icov_det[i])
                if sample_density[i,j] != sample_density[i,j]:
                    pdb.set_trace()
                if sample_density[i,j] == np.inf:
                    pdb.set_trace()
        density2 = np.zeros( (len(x_grid), len(y_grid)), dtype=np.int16)
        tree_xy = cKDTree(sample_xyzuvw[:,0:2])
        for x_ix, x in enumerate(x_grid):
            for y_ix, y in enumerate(y_grid):
                points = tree_xy.query_ball_point([x,y], smooth_pc)
                if len(points):
                    density2[y_ix, x_ix] = np.sum(sample_density[np.unravel_index(points, (ns, nsamp))])
    else:
        density2 = None

    print("Computing Density of stars...")
    density = np.zeros( (len(x_grid), len(y_grid)), dtype=np.int)
    tree_xy = cKDTree(sample_xyzuvw[:,0:2])
    for x_ix, x in enumerate(x_grid):
        for y_ix, y in enumerate(y_grid):
            points = tree_xy.query_ball_point([x,y], smooth_pc)
            p_ix = np.where(np.abs(sample_xyzuvw[points,2]) < plane_thickness)[0]
            #plane_points = points[p_ix]
            density[y_ix, x_ix] = len(p_ix)
  
    plt.figure(1)
    plt.clf()     
    #As we're plotting axes with negative in the bottom left corner, we need to reverse 
    #the y axis from python convention.
    plt.imshow(density[::-1,:], extent=[-axis_lim, axis_lim, -axis_lim, axis_lim])
    #for group, xyzuvw_group in zip(groups, xyzuvw_groups):
    for group, xyzuvw_group in zip(groups, xyzuvw_traceback_groups):
        #xplot = xyzuvw_group[0] + 1.023*target_time*xyzuvw_group[3]
        #yplot = xyzuvw_group[1] - 1.023*target_time*xyzuvw_group[4]
        xplot = xyzuvw_group[t_ix,0]
        yplot = xyzuvw_group[t_ix,1]
        if group=='Sun':
            plt.plot(xplot, yplot, 'x', color='orange')
        else:
            plt.plot(xplot, yplot, 'rx')
        plt.text(xplot+10, yplot, group)
    plt.axis([-axis_lim, axis_lim, -axis_lim, axis_lim])
    plt.title('{0:5.1f} Myr'.format(target_time))
    plt.xlabel('x (pc)')
    plt.ylabel('y (pc)')
    plt.colorbar()
    plt.pause(0.001)
    if savefigs:
        plt.savefig('imgs/d{0:03d}.png'.format(int(target_time)))
    
    if density2:
        plt.figure(2)
        plt.clf()     
        plt.imshow(np.sqrt(density2), extent=[np.min(x_grid), np.max(x_grid), np.min(y_grid), np.max(y_grid)])
        plt.title('{0:5.1f} Myr'.format(target_time))
        plt.xlabel('x (pc)')
        plt.ylabel('y (pc)')
        plt.colorbar()
        plt.pause(0.001)
        if savefigs:
            plt.savefig('imgs/g{0:03d}.png'.format(int(target_time)))
