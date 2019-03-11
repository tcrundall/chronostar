# coding: utf-8
from astropy.io import fits
hdul = fits.open("all_rvs_w_ok_plx.fits")
hdul[0].header
hdul[1].data.shape
hdul[1].data.columns
hdul[1].data['parallax_error'].median()
np.median(hdul[1].data['parallax_error'])
import numpy as np
np.median(hdul[1].data['parallax_error'])
np.median(hdul[1].data['pmra_error'])
np.median(hdul[1].data['pmdec_error'])
np.median(hdul[1].data['radial_velocity_error'])
np.std(hdul[1].data['radial_velocity_error'])
np.percentile(hdul[1].data['radial_velocity_error'], 80)
np.percentile(hdul[1].data['radial_velocity_error'], 50)
np.percentile(hdul[1].data['radial_velocity_error'], 84)
np.percentile(hdul[1].data['radial_velocity_error'], 16)
np.percentile(hdul[1].data['parallax_error'], 16)
np.percentile(hdul[1].data['parallax_error'], 50)
np.percentile(hdul[1].data['parallax_error'], 84)
np.std(hdul[1].data['parallax_error'], 84)
np.std(hdul[1].data['parallax_error'])
