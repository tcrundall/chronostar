#! /usr/bin/env python

import chronostar.traceback as tb
import chronostar.groupfitter as gf
import chronostar._overlap as ov
import numpy as np
import pdb
import pickle
try:
    import astropy.io.fits as pyfits
except:
    import pyfits

# TWA Hya init pos

twa_xyzuvw = np.array([12.49, -42.28, 21.55, -9.95, -17.91, -4.65])
twa_age    = 10
twa_params = list(twa_xyzuvw) + [1/5., 1/5., 1/5., 1/2., 0, 0, 0] + [twa_age]
twa_group = gf.Group(twa_params, 1.0)

#twa_origin = tb.traceback_group(twa_xyzuvw, twa_age)

# Nasty hacky way of checking if on Raijin
onRaijin = True
try:
    dummy = None
    pickle.dump(dummy, open("/short/kc5/dummy.pkl",'w'))
except:
    onRaijin = False

filename = "TGAS_traceback_165Myr_small.fits"
if onRaijin:
    location = "/short/kc5/"
else:
    location = "data/"

#table_infile = location + "Astrometry_with_RVs_250pc_100kms_lesscols.fits"

# Same table but with all columns
table_infile = location + "Astrometry_with_RVs_250pc_100kms.fits"
#table_infile = location + "Astrometry_with_RVs_subset2.fits"
#table_infile = location + filename
table = pyfits.getdata(table_infile)
# print(table.field('Notional Group'))

TWA_ixs = np.where(table['Notional Group'] == 'TWA')
TWA_9A_ix = np.where(table['Name1'] == 'TWA 9A')[0]

infile = location + filename
star_params = gf.read_stars(infile)

TWA_9A = star_params['stars'][TWA_9A_ix]
# twa_9a fields I'm interested in:
#  ra_adopt, dec_adopt, parallax_1, pmra_1, pmdec, pmra_error, pmdec_error
#  parallax_pmra_corr, parallax_pmdec_corr, ... check out traceback.py ln 272
#  for more details

nstars = star_params['stars'].size

star_mns, star_icovs, star_icov_dets = gf.interp_icov(twa_age, star_params)

overlaps = ov.get_overlaps(
    twa_group.icov, twa_group.mean, twa_group.icov_det, 
    star_icovs, star_mns, star_icov_dets, nstars
    )
twa_star_ixs = np.where(overlaps > np.percentile(overlaps, 99.9))
pdb.set_trace()
