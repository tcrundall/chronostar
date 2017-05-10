#! /usr/bin/env python

import chronostar.traceback as tb
import chronostar.groupfitter as gf
import chronostar._overlap as ov
import numpy as np
import pdb
import pickle
from csv import reader
from astropy.table import Table, Column
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

# Importing TWA astrometry from Donaldson16
def rahours_to_raDeg(hrs, mins, secs):
    return (hrs + mins/60. + secs/3600.)/24 * 360

def decdeg_to_degdec(degs, mins, secs):
    return degs + mins/60. + secs/3600.

def convert_ra(rahrs_str):
    elements_str = np.array(rahrs_str.split(' '))
    elements_flt = elements_str.astype(np.float)
    return rahours_to_raDeg(*elements_flt)

def convert_dec(decdeg_str):
    elements_str = np.array(decdeg_str.split(' '))
    elements_flt = elements_str.astype(np.float)
    return decdeg_to_degdec(*elements_flt)

# Taken from www.bdnyc.org/2012/10/decimal-deg-to-hms/
def HMS2deg(ra='', dec=''):
    RA, DEC, rs, ds = '', '', 1, 1
    if dec:
        D, M, S = [float(i) for i in dec.split()]
        if str(D)[0] == '-':
            ds, D = -1, abs(D)
        deg = D + (M/60) + (S/3600)
        DEC = '{0}'.format(deg*ds)

    if ra:
        H, M, S = [float(i) for i in ra.split()]
        if str(H)[0] == '-':
            rs, H = -1, abs(H)
        deg = (H*15) + (M/4) + (S/240)
        RA = '{0}'.format(deg*rs)

    if ra and dec:
        return (RA, DEC)
    else:
        return RA or DEC

infile = open(location + 'Donaldson16_TWA_astrometry.csv', 'r')
data = []
for line in reader(infile):
    data += [line]
data = np.array(data)

nTWAstars = data.shape[0]
RA  = np.zeros(nTWAstars)
DEC = np.zeros(nTWAstars)

# converting ra and dec measurments to decimal
for i in range(nTWAstars):
    RA[i], DEC[i] = HMS2deg(data[i][1], data[i][2])

Plx, e_Plx, pmDE, e_pmDE, pmRA, e_pmRA =\
    data[:,3], data[:,4], data[:,11], data[:,12], data[:,9], data[:,10]
RV, e_RV = data[:,6], data[:,7]

# make a dectionary
stars = {}
stars['Name']  = data[:,0]
stars['RAdeg'] = RA
stars['DEdeg'] = DEC
stars['Plx']   = Plx
stars['e_Plx'] = e_Plx
stars['RV']    = RV
stars['e_RV']  = e_RV
stars['pmRA']  = pmRA
stars['e_pmRA']= e_pmRA
stars['pmDE']  = pmDE
stars['e_pmDE']= e_pmDE

t = Table(
    [data[:,0],
     RA.astype(np.float),    
     DEC.astype(np.float),
     Plx.astype(np.float),
     e_Plx.astype(np.float),
     RV.astype(np.float),
     e_RV.astype(np.float),
     pmRA.astype(np.float),
     e_pmRA.astype(np.float),
     pmDE.astype(np.float),
     e_pmDE.astype(np.float)],
    names=('Name', 'RAdeg','DEdeg','Plx','e_Plx','RV','e_RV',
           'pmRA','e_pmRA','pmDE','e_pmDE')
    )
times = np.linspace(0,15,40)
xyzuvw = tb.traceback(t,times,savefile=location + 'TWA_traceback.pkl')

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
