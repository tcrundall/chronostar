# coding: utf-8
get_ipython().magic(u'run ipython_primer.py')
import numpy as np
import chronostar.compfitter as gf
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
import chronostar.compfitter as gf
star_pars = gf.loadXYZUVW("../data/gaia_dr2_bp_xyzuvw.fits")
star_pars['xyzuvw']
import matplotlib.pyplot as plt
plt.plot(star_pars['xyzuvw'][:,0], star_pars['xyzuvw'][:,1], '.')
plt.show()
plt.plot(star_pars['xyzuvw'][:,3], star_pars['xyzuvw'][:,4], '.')
plt.show()
gf.fit_comp(xyzuvw_dict=star_pars, plot_it=True, plot_dir='temp_plots')
