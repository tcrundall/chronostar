"""
measurer module

Replicates the measurement of synthetic stars with precisely known XYZUVW
values by converting into RA, DEC, parallax, proper motion, radial velocity
with
appropriate errors.
"""

from __future__ import print_function, division

from astropy.table import Table
import numpy as np
import pickle

import coordinate as cc

# BRIGHT STARS
GERROR = {
    'e_Plx':0.04, #e_Plx [mas]
    'e_RV' :0.3,  #e_RV [km/s]
    'e_pm' :0.06, #e_pm [mas/yr]
}
#GERROR = {
#    'e_Plx':0.04, #e_Plx [mas]
#    'e_RV' :0.6,  #e_RV [km/s]
#    'e_pm' :0.06, #e_pm [mas/yr]
#}
# 2017 'observational error'
#GERROR = {
#    'e_Plx':0.5, #e_Plx [mas]
#    'e_RV' :1.,  #e_RV [km/s]
#    'e_pm' :10., #e_pm [mas/yr]
#}
## FAINT STARS
#GERROR = {
#    'e_Plx':0.7, #e_Plx [mas]
#    'e_RV' :1.2,  #e_RV [km/s]
#    'e_pm' :1.2, #e_pm [mas/yr]
#}
## MIDDLING STARS
#GERROR = {
#    'e_Plx':0.1, #e_Plx [mas]
#    'e_RV' :0.6,  #e_RV [km/s]
#    'e_pm' :0.2, #e_pm [mas/yr]
#}

def convertArrayToTable(astros, errors):
    """Utility function to generate table"""
    nstars = astros.shape[0]
    t = Table(
        [
            np.arange(nstars),
            astros[:,0],
            astros[:,1],
            astros[:,2],
            errors[:,2],
            astros[:,3],
            errors[:,3],
            astros[:,4],
            errors[:,4],
            astros[:,5],
            errors[:,5],
        ],
        names=('name', 'radeg','dedeg','plx','e_plx',
               'pmra','e_pmra','pmde','e_pmde','rv','e_rv'),
    )
    return t

def convertAstroArrayToTable(astros):
    t = Table(
        rows = astros,
        #[
        #    astros[:,0],
        #    astros[:,1],
        #    astros[:,2],
        #    astros[:,3],
        #    astros[:,4],
        #    astros[:,5],
        #    astros[:,6],
        #    astros[:,7],
        #    astros[:,8],
        #    astros[:,9],
        #    astros[:,10],
        #],
        names=('name', 'radeg','dedeg','plx','e_plx',
               'pmra','e_pmra','pmde','e_pmde','rv','e_rv'),
        dtype = ('S20',
                'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'),
    )
    return t

def convertTableToArray(star_table):
    nstars = star_table['radeg'].shape[0]
    measured_vals = np.vstack((
        star_table['radeg'],
        star_table['dedeg'],
        star_table['plx'],
        star_table['pmra'],
        star_table['pmde'],
        star_table['rv'],
    )).T

    errors = np.vstack((
        np.zeros(nstars),
        np.zeros(nstars),
        star_table['e_plx'],
        star_table['e_pmra'],
        star_table['e_pmde'],
        star_table['e_rv'],
    )).T
    return measured_vals, errors


def measureXYZUVW(xyzuvws, error_frac, savefile=''):
    """
    Replicates the measurement of synthetic stars. Converts XYZUVW to radec..

    Parameters
    ----------
    xyzuvws : ([nstars, 6] float array) A list of stars in rh cartesian
        coordinate system, centred on and co-rotating with the local standard
        of rest
        [pc, pc, pc, km/s, km/s, km/s]
    error_frac : (0 - inf float) Parametrisation of Gaia-like uncertainty. 0.
        is perfect precision, 1.0 is simplified best Gaia uncertainty.
        Gaia uncertainty is taken to be: e_plx=0.04 mas, e_rv=0.3 km/s,
        e_pm=0.06 mas/yr
    savefile : string {''}
        if not empty, the astrometry table will be saved to the given
        file name

    Returns
    -------
    real_astros : ([nstars, 6] float array) List of stars in measurements with
        incorporated error
    """
    errors = np.array([
        0., 0., GERROR['e_Plx'], GERROR['e_pm'], GERROR['e_pm'], GERROR['e_RV']
    ])
    nstars = xyzuvws.shape[0]
    astros = cc.convertManyLSRXYZUVWToAstrometry(xyzuvws)

    raw_errors = error_frac * np.tile(errors, (nstars, 1))
    random_errors = raw_errors * np.random.randn(*raw_errors.shape)

    astros_w_errs = astros + random_errors
    astrometry_table = convertArrayToTable(astros_w_errs, raw_errors)
    #if as_table:
    #    astros_w_errs = convertArrayToTable(astros_w_errs, raw_errors)


    if savefile:
        astrometry_table.write(savefile, format='ascii', overwrite=True)

    return astrometry_table

