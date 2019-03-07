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

# BASED ON MEDIAN ERRORS OF ALL GAIA STARS WITH RVS
# AND <20% PARALLAX ERROR
GERROR = {
    'e_Plx':0.035, #e_Plx [mas]
    'e_RV' :1.0,  #e_RV [km/s]
    'e_pm' :0.05, #e_pm [mas/yr]
}

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


