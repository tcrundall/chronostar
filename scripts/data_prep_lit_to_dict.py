"""
Take (excerpt of) the lit compiled RV table and convert into standardised
format and
"""
from __future__ import print_function, division

from astropy.io import fits
from astropy.table import Table, Column
import csv
import matplotlib.pyplot as plt
import numpy as np
import logging
import pdb
import pickle
import sys

sys.path.insert(0, '..')

import chronostar.retired.groupfitter as rgf
import chronostar.measurer as ms
import chronostar.converter as cv
import chronostar.datatool as dt

#original_tb_file = "../data/bp_TGAS2_traceback_save.pkl"
data_file = "../data/bpmg_cand_w_gaia_dr2_astrometry_comb_binars.csv"
#astro_file = "../data/bp_astro.dat"
astr_file = "../data/bpmg_cand_w_gaia_dr2_astrometry_comb_binars.fits"
xyzuvw_file = "../data/bpmg_cand_w_gaia_dr2_astrometry_comb_binars_xyzuvw.fits"

main_rv_name = "RV"
main_erv_name = "RV error"
name_name = "Name1"
gaia_ra_name = "ra" # this column is start of the gaia contiguous data block
gaia_dec_name = "dec"
gaia_start_name = "parallax" # this column is start of the gaia contiguous data block
gaia_end_name = "pmra_pmdec_corr"


with open(data_file, 'r') as fp:
    rd = csv.reader(fp)

    header = rd.next()
    data_str = np.zeros((0,len(header))).astype(np.str)
    for row in rd:
        data_str = np.vstack((data_str, row))



main_rv_ix = header.index(main_rv_name)
main_erv_ix = header.index(main_erv_name)
name_ix = header.index(name_name)
gaia_ra_ix = header.index(gaia_ra_name)
gaia_dec_ix = header.index(gaia_dec_name)
gaia_start_ix = header.index(gaia_start_name)
gaia_end_ix = header.index(gaia_end_name) + 1
NSTARS = data_str.shape[0]

# data_ordered = np.vstack((
#     data_str[:,name_ix],
#     data_str[:,gaia_ra_ix],
#     data_str[:,gaia_dec_ix],
#     data_str[:,gaia_start_ix:gaia_end_ix].T,
#     data_str[:,main_rv_ix],
#     data_str[:,main_erv_ix],
# ))
# data_ordered = data_ordered.T

gaia_file = "../data/all_rvs_w_ok_plx.fits"
gaia_master_table = Table.read(gaia_file)
master_dtype = gaia_master_table.dtype

new_table = Table(data=np.zeros(NSTARS, dtype=master_dtype))
for col in gaia_master_table.columns:
    try:
        new_table[col] = data_str[:,header.index(col)].astype(
            master_dtype.fields[col][0]
        )
    except ValueError:
        print("Column {} not present or can't convert value".format(col))

# manually insert RV
new_table['radial_velocity'] = data_str[:,main_rv_ix].astype(
    master_dtype.fields['radial_velocity'][0]
)
new_table['radial_velocity_error'] = data_str[:,main_erv_ix].astype(
    master_dtype.fields['radial_velocity_error'][0]
)

new_table.write(astr_file, format='fits', overwrite=True)

xyzuvw_dict = dt.convertGaiaToXYZUVWDict(astr_file, return_dict=True)

#    hdu = fits.BinTableHDU(data=hdul[1].data[mask])
#    new_hdul = fits.HDUList([primary_hdu, hdu])
#    new_hdul.writeto(filename, overwrite=True)


#plt.plot(xyzuvw_dict['xyzuvw'][:,0], xyzuvw_dict['xyzuvw'][:,1], '.')
#plt.show()
#
# print("Done")



