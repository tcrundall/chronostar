"""This is a test script for trying to load """

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.io.fits as pyfits
import pylab as p
import chronostar.traceback as traceback
import chronostar.fit_group as fit_group
#plt.ion()

trace_it_back = True
fit_the_group = False
n_times = 2
max_time = 18.924

#t=Table.read('Astrometry_with_RVs_subset.fits')
#t['Dist'] = 10e-3/t['parallax_1']
#t = t[(t['Dist'] < 0.05) & (t['Dist'] > 0)]
#vec_wd = np.vectorize(traceback.withindist)
#t = t[vec_wd((t['ra_adopt']),(t['dec_adopt']),(t['Dist']),0.02, 86.82119870, -51.06651143, 0.01944)]
#t=Table.read('data/betaPic_RV_Check2.csv')

#Limit outselves to stars with parallaxes smaller than 250pc, uncertainties smaller than 20%
#and radial velocities smaller than 100km/s. This gives 65,853 stars.
t = pyfits.getdata("/Users/mireland/Google Drive/chronostar_catalogs/Astrometry_with_RVs_subset2.fits",1)
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
    tb.traceback(times,xoffset=xoffset, yoffset=yoffset, axis_range=axis_range, dims=dims,plotit=False,savefile="results/bp_TGAS2_traceback_save.pkl")

if fit_the_group:
    star_params = fit_group.read_stars("results/bp_TGAS1_traceback_save.pkl")
    
    beta_pic_group = np.array([-6.574, 66.560, 23.436, -1.327,-11.427, -6.527,\
        10.045, 10.319, 12.334,  0.762,  0.932,  0.735,  0.846, 20.589])
 
    beta_pic_group = np.array([-6.574, 66.560, 23.436, -1.327,-11.427, 0,\
     10.045, 10.319, 12.334,  5,  0.932,  0.735,  0.846, 20.589])

    ol_swig = fit_group.lnprob_one_group(beta_pic_group, star_params, use_swig=True, return_overlaps=True)
    ol_old  = fit_group.lnprob_one_group(beta_pic_group, star_params, use_swig=False, return_overlaps=True)

#    fitted_group = fit_group.fit_one_group(star_params, init_mod=beta_pic_group,\
#        nwalkers=30,nchain=100,nburn=20, return_sampler=False,pool=None,\
#        init_sdev = np.array([1,1,1,1,1,1,1,1,1,.01,.01,.01,.1,1]), background_density=2e-12, use_swig=False, \
#        plotit=True)
