"""
measurer module

Replicates the measurement of synthetic stars with precisely known XYZUVW values
by converting into RA, DEC, parallax, proper motion, radial velocity with
appropriate errors.
"""

from __future__ import print_function, division

import numpy as np
import coordinate as cc

GERROR = {
    'e_Plx':0.04, #e_Plx [mas]
    'e_RV' :0.3,  #e_RV [km/s]
    'e_pm' :0.06, #e_pm [mas/yr]
}

def measureXYZUVW(xyzuvws, error_frac):
    """
    Replicates the measurement of synthetic stars

    Parameters
    ----------
    xyzuvws : ([nstars, 6] float array) A list of stars in rh cartesian
        coordinate system, centred on and co-rotating with the local standard
        of rest
        [pc, pc, pc, km/s, km/s, km/s]
    error_frac : (0 - inf float) Parametrisation of Gaia-like uncertainty. 0. is
        perfect precision, 1.0 is simplified best Gaia uncertainty.
        Gaia uncertainty is taken to be: e_plx=0.04 mas, e_rv=0.3 km/s,
        e_pm=0.06 mas/yr

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

    raw_errors = np.tile(errors, (nstars, 1))
    random_errors = error_frac * raw_errors * np.random.randn(*raw_errors.shape)

    astros_w_errs = astros + random_errors

    return astros_w_errs

