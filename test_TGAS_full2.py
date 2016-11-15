"""This is a test script for trying to use the full TGAS set of stars.

import chronostar
star_params = chronostar.fit_group.read_stars('results/bp_TGAS2_traceback_save.pkl')
beta_pic_group = np.array([ -0.908, 60.998, 27.105, -0.651,-11.470, -0.148,  8.055,  4.645,  8.221,  0.655,  0.792,  0.911,  0.843])
star_probs = chronostar.fit_group.lnprob_one_group(beta_pic_group, star_params, return_overlaps=True, t_ix=1)
possible_members = np.where( (np.log10(star_probs) > -18) * (np.log10(star_probs) < 0) )[0]
table_subset = t[possible_members]   
pyfits.writeto('/Users/mireland/Google Drive/chronostar_catalogs/Astrometry_with_RVs_possible_bpic.fits', table_subset)

"""

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.io.fits as pyfits
import pylab as p
import chronostar.traceback as traceback
import chronostar.fit_group as fit_group
from emcee.utils import MPIPool
import sys
import pickle
#plt.ion()

trace_it_back = False 
fit_the_group = True
n_times = 2
max_time = 18.924

use_bpic_subset=True

n_times = 41
max_time = 40

n_times=56
max_time=165
pklfile="TGAS_traceback_165Myr.pkl"
pklfile = "bp_TGAS2_traceback_save.pkl"
ddir = "data/"
#ddir = ""/Users/mireland/Google Drive/chronostar_catalogs/"
#----

#t=Table.read('Astrometry_with_RVs_subset.fits')
#t['Dist'] = 10e-3/t['parallax_1']
#t = t[(t['Dist'] < 0.05) & (t['Dist'] > 0)]
#vec_wd = np.vectorize(traceback.withindist)
#t = t[vec_wd((t['ra_adopt']),(t['dec_adopt']),(t['Dist']),0.02, 86.82119870, -51.06651143, 0.01944)]
#t=Table.read('data/betaPic_RV_Check2.csv')

#Limit outselves to stars with parallaxes smaller than 250pc, uncertainties smaller than 20%
#and radial velocities smaller than 100km/s. This gives 65,853 stars.
if use_bpic_subset:
    t = pyfits.getdata(ddir + "Astrometry_with_RVs_possible_bpic.fits",1)
else:
    t = pyfits.getdata(ddir + "Astrometry_with_RVs_subset2.fits",1)
good_par_sig = np.logical_or(t['parallax_1'] > 5*t['parallax_error'], t['Plx'] > 5*t['e_Plx'])
good_rvs =np.abs(t['rv_adopt']) < 100
good_par = np.logical_or(t['parallax_1'] > 4, t['Plx'] > 4)
good = np.logical_and(np.logical_and(good_par_sig,good_par),good_rvs)
t = t[good]



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
    tb.traceback(times,xoffset=xoffset, yoffset=yoffset, axis_range=axis_range, dims=dims,plotit=False,savefile="results/"+pklfile)

if fit_the_group:
    star_params = fit_group.read_stars("results/"+pklfile)
    
    beta_pic_group = np.array([ -0.908, 60.998, 27.105, -0.651,-11.470, -0.148, \
      8.055,  4.645,  8.221,  0.655,  0.792,  0.911,  0.843, 18.924])
 
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
        init_sdev = np.array([1,1,1,1,1,1,1,1,1,.01,.01,.01,.1,.1]), background_density=1e-6, use_swig=True, \
        plotit=False)
    
    if using_mpi:
        # Close the processes.
        pool.close()

    #print("Autocorrelation lengths: ")
    #print(sampler.get_autocorr_time(c=2.5))
    pickle.dump((sampler.chain[:,-1,:], sampler.lnprobability[:,-1]), open("betaPic_sampler_end_m06.pkl",'w')) 
    print("Autocorrelation lengths: ")
    print(sampler.get_autocorr_time(c=2.0))
