#! /usr/bin/env python
"""This is a test script for tracing back beta Pictoris stars.
This script is redundant. Code can be incorporated into other files.
"""

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import pylab as p
import chronostar.traceback as traceback
import chronostar.fit_group as fit_group
from emcee.utils import MPIPool
import sys
import pickle
#plt.ion()

trace_it_back = False
fit_the_group = True
n_times = 31
max_time = 30

#t=Table.read('Astrometry_with_RVs_subset.fits')
#t['Dist'] = 10e-3/t['parallax_1']
#t = t[(t['Dist'] < 0.05) & (t['Dist'] > 0)]
#vec_wd = np.vectorize(traceback.withindist)
#t = t[vec_wd((t['ra_adopt']),(t['dec_adopt']),(t['Dist']),0.02, 86.82119870, -51.06651143, 0.01944)]
t=Table.read('data/betaPic_RV_Check2.csv')

#Which dimensions do we plot? 0=X, 1=Y, 2=Z
dims = [0,1]
dim1=dims[0]
dim2=dims[1]
xoffset = np.zeros(len(t))
yoffset = np.zeros(len(t))

#Some hardwired plotting options.
if (dims[0]==0) & (dims[1]==1):
    #yoffset[0:10] = [6,-8,-6,2,0,-4,0,0,0,-4]
    #yoffset[10:] = [0,-8,0,0,6,-6,0,0,0]
    #xoffset[10:] = [0,-4,0,0,-15,-10,0,0,-20]
    axis_range = [-70,60,-40,120]

if (dims[0]==1) & (dims[1]==2):
    axis_range = [-40,120,-30,100]
    #text_ix = [0,1,4,7]
    #xoffset[7]=-15
    
times = np.linspace(0,max_time, n_times)

if trace_it_back:
    tb = traceback.TraceBack(t)
    tb.traceback(times,xoffset=xoffset, yoffset=yoffset, axis_range=axis_range, dims=dims,plotit=True,savefile="results/bp_TGAS1_traceback_save.pkl")

if fit_the_group:
    star_params = fit_group.read_stars("results/bp_TGAS1_traceback_save.pkl")
    
    #Original
    beta_pic_group = np.array([-6.574, 66.560, 23.436, -1.327,-11.427, -6.527,\
        10.045, 10.319, 12.334,  0.762,  0.932,  0.735,  0.846, 20.589])
    #Widened
    beta_pic_group = np.array([-6.574, 66.560, 23.436, -1.327,-11.427, 0,\
     10.045, 10.319, 12.334,  5,  0.932,  0.735,  0.846, 20.589])
    #After one successful fit.
    beta_pic_group = np.array([ -0.908, 60.998, 27.105, -0.651,-11.470, -0.148,  8.055,  4.645,  8.221,  0.655,  0.792,  0.911,  0.843, 18.924])
    beta_pic_group = np.array([ -1.96 ,  60.281,  25.242,   0.359, -11.864,  -0.175,   5.516,4.497,   7.993,   0.848,   0.51 ,   0.776,   0.765,  18.05 ])

    ol_swig = fit_group.lnprob_one_group(beta_pic_group, star_params, use_swig=True, return_overlaps=True)
    ol_old  = fit_group.lnprob_one_group(beta_pic_group, star_params, use_swig=False, return_overlaps=True)

    using_mpi = True
    try:
        # Initialize the MPI-based pool used for parallelization.
        pool = MPIPool()
    except:
        print("Either MPI doesn't seem to be installed or you aren't running with MPI... ")
        using_mpi = False
        pool=None
    
    if using_mpi:
        if not pool.is_master():
            # Wait for instructions from the master process.
            pool.wait()
            sys.exit(0)
    else:
        print("MPI available for this code! - call this with e.g. mpirun -np 16 python test_betapic_TGAS.py")

    sampler = fit_group.fit_one_group(star_params, init_mod=beta_pic_group,\
        nwalkers=30,nchain=10000,nburn=1000, return_sampler=True,pool=pool,\
        init_sdev = np.array([1,1,1,1,1,1,1,1,1,.01,.01,.01,.1,.1])*0.1, background_density=1e-9, use_swig=True, \
        plotit=True)
    
    if using_mpi:
        # Close the processes.
        pool.close()

    #print("Autocorrelation lengths: ")
    #print(sampler.get_autocorr_time(c=2.5))
    pickle.dump((sampler.chain, sampler.lnprobability), open("betaPic_sampler_test.pkl",'w'))  
