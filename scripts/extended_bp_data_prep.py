# coding: utf-8
get_ipython().magic(u'run ipython_primer.py')
import numpy as np
import chronostar.groupfitter as gf
xyzuvw_dict = gf.load("../data/gaia_dr2_ok_plx_xyzuvw.fits.gz.fits")
xyzuvw_dict = gf.loadXYZUVW ("../data/gaia_dr2_ok_plx_xyzuvw.fits.gz.fits")
xyzuvw_dict.keys()
xyzuvw_dict['xyzuvw'].shape
xyzuvw_dict['xyzuvw_cov'].shape
get_ipython().magic(u'cat log_gaia_converter.log')
get_ipython().magic(u'cat log_gaia_converter.log')
get_ipython().magic(u'cat log_gaia_converter.log')
get_ipython().magic(u'cat log_gaia_converter.log')
get_ipython().magic(u'cat log_gaia_converter.log')
get_ipython().magic(u'cat log_gaia_converter.log')
get_ipython().magic(u'cat log_gaia_converter.log')
get_ipython().magic(u'cat log_gaia_converter.log')
get_ipython().magic(u'cat log_gaia_converter.log')
get_ipython().magic(u'cat log_gaia_converter.log')
get_ipython().magic(u'cat log_gaia_converter.log')
get_ipython().magic(u'cat log_gaia_converter.log')
get_ipython().magic(u'cat log_gaia_converter.log')
get_ipython().magic(u'cat log_gaia_converter.log')
get_ipython().magic(u'ls ')
get_ipython().magic(u'rm nohup.out')
get_ipython().magic(u'rm log_gaia_converter.log')
get_ipython().magic(u'ls ')
6000*5
6000*5/60
6000*5/60/60
8*60
6000*5./60./60.
6370000./35000 * 3.5
(6370000./35000 * 3.5)/60
gaia_bp_astro_file = "../data/gaia_dr2_bp_astro.fits"
hdul = fits.open(gaia_bp_astro_file)
from astropy.io import fits
hdul = fits.open(gaia_bp_astro_file)
from scripts.retired import gaia_converter as gc

hdul[1].data.shape
hdul[1].data[0]
hdul[1].header
means, covs = gc.convertManyRecToArray(hdul[1].data)
astr_dict = {'astr_mns':means, 'astr_covs':covs}
astr_dict['astr_mns'].shape
astr_dict['astr_covs'].shape
np.sqrt(astr_dict['astr_covs'][0])
hdul[1].data[0]['error_pmra']
hdul[1].data[0]['pmra_error']
hdul[1].data[0]['parralax_error']
hdul[1].data[0]['parallax_error']
np.sqrt(astr_dict['astr_covs'][0])
import chronostar.retired2.converter as cv
cv.convertMeasurementsToCartesian(astr_dict=astr_dict, savefile="../data/gaia_dr2_bp_xyzuvw.fits")
import chronostar.groupfitter as gf
star_pars = gf.loadXYZUVW("../data/gaia_dr2_bp_xyzuvw.fits")
star_pars['xyzuvw']
import matplotlib.pyplot as plt
plt.plot(star_pars['xyzuvw'][:,0], star_pars['xyzuvw'][:,1], '.')
plt.show()
plt.plot(star_pars['xyzuvw'][:,3], star_pars['xyzuvw'][:,4], '.')
plt.show()
gf.fit_comp(xyzuvw_dict=star_pars, plot_it=True, plot_dir='temp_plots')
get_ipython().magic(u'save gaia_dr2_bp_data_generation.py 1-62')
xyzuvw_dict['xyzuvw'].shape
star_pars
star_pars['xyzuvw'].shae
star_pars['xyzuvw'].shape
plt.plot(star_pars['xyzuvw'][0], star_pars['xyzuvw'][3], '.')
plt.show()
plt.plot(star_pars['xyzuvw'][:,0], star_pars['xyzuvw'][:,3], '.')
plt.show()
np.where(star_pars['xyzuvw'][:,3] > -10)
np.where((star_pars['xyzuvw'][:,3] > -10) & star_pars['xyzuvw'][:3] < 20)
np.where((star_pars['xyzuvw'][:,3] > -10) & (star_pars['xyzuvw'][:3] < 20))
np.where((star_pars['xyzuvw'][:,3] > -10) & (star_pars['xyzuvw'][:,3] < 20))
bp_membs = np.where((star_pars['xyzuvw'][:,3] > -10) & (star_pars['xyzuvw'][:,3] < 20))
bp_membs
plt.plot(star_pars['xyzuvw'][bp_membs,0], star_pars['xyzuvw'][bp_membs,3], '.')
plt.show()
plt.plot(star_pars['xyzuvw'][:,0], star_pars['xyzuvw'][:,3], 'b.')
plt.plot(star_pars['xyzuvw'][bp_membs,0], star_pars['xyzuvw'][bp_membs,3], 'r.')
plt.show()
f, axes = plt.subplots(1,3,figsize=(15,8))
axes[0,0].plot(star_pars['xyzuvw'][bp_membs,0],
star_pars['xyzuvw'][bp_membs,3], 'r.')
axes.shape
axes[0].plot(star_pars['xyzuvw'][bp_membs,0],
star_pars['xyzuvw'][bp_membs,3], 'r.')
axes[1].plot(star_pars['xyzuvw'][bp_membs,1],
star_pars['xyzuvw'][bp_membs,4], 'r.')
axes[2].plot(star_pars['xyzuvw'][bp_membs,2],
star_pars['xyzuvw'][bp_membs,5], 'r.')
plt.savefig("bpmg-pos-vel-corrs.pdf")
np.where(star_pars['xyzuvw'][:,4] < -30)
bp_membs
star_pars['xyzuvw'][6]
bp_membs = np.where((star_pars['xyzuvw'][:,3] > -10) & (star_pars['xyzuvw'][:,3] < 10))
plt.clf()
f, axes = plt.subplots(1,3,figsize=(15,5))
axes[0].plot(star_pars['xyzuvw'][bp_membs,0],
star_pars['xyzuvw'][bp_membs,3], 'r.')
axes[1].plot(star_pars['xyzuvw'][bp_membs,1],
star_pars['xyzuvw'][bp_membs,4], 'r.')
axes[2].plot(star_pars['xyzuvw'][bp_membs,2],
star_pars['xyzuvw'][bp_membs,5], 'r.')
plt.savefig("bpmg-pos-vel-corrs.pdf")
bpmg_memb
bp_membs
demo_membs = (np.array([ 4,  9, 10, 12, 13, 14, 15, 16, 17, 18, 19]),)
demo_membs
star_pars['xyzuvw'][demo_membs]
z = np.zeros
z[bp_membs] = 1
z[bp_membs] 
z = np.zeros()
nstars = star_pars['xyzuvw'].shape[0]
nstars
z = np.zeros(nstars)
z[bp_membs] = 1.
z
np.save("bp_init_memb.npy", z)
star_pars['xyzuvw'][:,2]
star_pars['xyzuvw'][bp_membs,2]
star_pars['xyzuvw'][bp_membs,2].mean
star_pars['xyzuvw'][bp_membs,2].mean()
