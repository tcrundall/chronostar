"""This is a test script for trying to use the full TGAS set of stars.

import chronostar
star_params = chronostar.fit_group.read_stars('results/LOST_FILE.pkl')
beta_pic_group = np.array([ -0.908, 60.998, 27.105, -0.651,-11.470, -0.148,  8.055,  4.645,  8.221,  0.655,  0.792,  0.911,  0.843])
star_probs = chronostar.fit_group.lnprob_one_group(beta_pic_group, star_params, return_overlaps=True, t_ix=1)
possible_members = np.where( (np.log10(star_probs) > -18) * (np.log10(star_probs) < 0) )[0]
table_subset = t[possible_members]   
pyfits.writeto('/Users/mireland/Google Drive/chronostar_catalogs/Astrometry_with_RVs_possible_bpic.fits', table_subset)

*** Trimming down the file ***
stars,times,xyzuvw,xyzuvw_cov = pickle.load(open('TGAS_traceback_165Myr.pkl'))
new_stars = pyfits.getdata('../data/Astrometry_with_RVs_250pc_100kms_lesscols.fits',1)
good = stars['rv_adopt_error'] < 10
good_ix = np.where(good)[0]

*** Saving our MASTER traceback ***
hdu0 = pyfits.PrimaryHDU()
hdu1 = pyfits.BinTableHDU(new_stars)
hdu2 = pyfits.ImageHDU(times)
hdu3 = pyfits.ImageHDU(xyzuvw)
hdu4 = pyfits.ImageHDU(xyzuvw_cov)
hdulist = pyfits.HDUList([hdu0,hdu1,hdu2,hdu3,hdu4])
hdulist.writeto('TGAS_traceback_165Myr_small.fits', clobber=True)

"""

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

use_bpic_subset=False

n_times = 41
max_time = 40

n_times=56
max_time=165
pklfile="TGAS_traceback_165Myr_delete.fits"
ddir = "data/"
ddir = "/Users/mireland/Google Drive/chronostar_catalogs/"
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
    t = pyfits.getdata(ddir + "Astrometry_with_RVs_250pc_100kms_lesscols.fits",1)
#    t = pyfits.getdata(ddir + "Astrometry_with_RVs_subset2.fits",1)

print("{0:d} stars before trimming...".format(len(t)))
good_par_sig = np.logical_or(t['parallax_1'] > 5*t['parallax_error'], t['Plx'] > 5*t['e_Plx']) #<20% parallax uncertainty.
good_rv_sig =t['rv_adopt_error'] < 10   #< 10km/s RV uncertainty
good_rvs =np.abs(t['rv_adopt']) < 100   #<100m/s RV, i.e. thin disk.    
good_par = np.logical_or(t['parallax_1'] > 4, t['Plx'] > 4) #<250pc from the sun
good = good_par_sig * good_rv_sig * good_par * good_rvs
t = t[good]
print("{0:d} stars after trimming...".format(len(t)))

#pyfits.writeto('/Users/mireland/Google Drive/chronostar_catalogs/Astrometry_with_RVs_250pc_100kms.fits', t)

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
