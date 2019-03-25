from astropy.io import fits
import numpy as np

## generally, if they don't have one, they won't have the others
#no_plxpmradecs = np.where((hdul[1].data['pmdec'] != hdul[1].data['pmdec']) |
#                       (hdul[1].data['pmra'] != hdul[1].data['pmra']) |
#                       (hdul[1].data['parallax'] != hdul[1].data['parallax']) 
#                       #(hdul[1].data['pmra'] != hdul[1].data['pmra']) |
#                      )
#
#no_plx = np.where(hdul[1].data['parallax'] != hdul[1].data['parallax'])

with fits.open('all_rvs.fits') as hdul:
    primary_hdu = fits.PrimaryHDU(header=hdul[1].header)
    has_ok_plx = np.where(
        hdul[1].data['parallax'] / hdul[1].data['parallax_error'] > 5.
    )
    hdu = fits.BinTableHDU(data=hdul[1].data[has_ok_plx])
    new_hdul = fits.HDUList([primary_hdu, hdu])
    new_hdul.writeto('all_rvs_w_ok_plx.fits')


