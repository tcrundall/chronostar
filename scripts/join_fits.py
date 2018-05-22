from astropy.io import fits

file_1 = 'all_non_pos_rv-result.fits.gz'
file_2 = 'all_pos_rv-result.fits.gz'

with fits.open(file_1) as hdul1:
    primary_hdu = fits.PrimaryHDU(header=hdul1[1].header)
    with fits.open(file_2) as hdul2:
        nrows1 = hdul1[1].data.shape[0]
        nrows2 = hdul2[1].data.shape[0]
        nrows = nrows1 + nrows2
        hdu = fits.BinTableHDU.from_columns(hdul1[1].columns, nrows=nrows)
        for colname in hdul1[1].columns.names:
            hdu.data[colname][nrows1:] = hdul2[1].data[colname]

hdul = fits.HDUList([primary_hdu, hdu])
hdul.writeto('all_rvs.fits')

hdul.close()

