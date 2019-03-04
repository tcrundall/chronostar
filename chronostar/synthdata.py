"""
synthesiser

Used to generate realistic data for one (or many) synthetic association(s)
with multiple starburst events along with a background as desired.

From a parametrised gaussian distribution, generate the starting
XYZUVW values for a given number of stars

TODO: accommodate multiple groups
"""

from __future__ import print_function, division

from astropy.table import Table, Column, vstack
import logging
import numpy as np

from chronostar import coordinate
from chronostar.component import Component
from chronostar import traceorbit


class SynthData():
    # BASED ON MEDIAN ERRORS OF ALL GAIA STARS WITH RVS
    # AND <20% PARALLAX ERROR
    GERROR = {
        'ra_error': 1e-6, #deg
        'dec_error': 1e-6, #deg
        'parallax_error': 0.035,  # e_Plx [mas]
        'radial_velocity_error': 1.0,  # e_RV [km/s]
        'pmra_error': 0.05,  # e_pm [mas/yr]
        'pmdec_error': 0.05,  # e_pm [mas/yr]
    }

    DEFAULT_ASTR_COLNAMES = (
        'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity',
    )

    DEFAULT_NAMES = (
        'name', 'component', 'age',
        'x0', 'y0', 'z0', 'u0', 'v0', 'w0',
        'x_now', 'y_now', 'z_now', 'u_now', 'v_now', 'w_now',
        'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error',
        'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
        'radial_velocity', 'radial_velocity_error',
    )

    DEFAULT_DTYPES = tuple(['S20', 'S2']
                           + (len(DEFAULT_NAMES)-2) * ['float64'])

    def __init__(self, pars, starcounts, measurement_error=1.0,
                 comp_forms='sphere', savedir=None, tablefilename=None):
        """
        Generates a set of astrometry data based on multiple star bursts with
        simple, Gaussian origins.

        TODO: allow for list of Components as input
        """
        # Tidying input and applying some quality checks
        self.pars = np.array(pars)
        assert len(self.pars.shape) == 2, 'pars must be a 2 dim array'
        self.ncomps = pars.shape[0]
        assert len(starcounts) == self.ncomps,\
            'starcounts must be same length as pars dimension'
        self.starcounts = starcounts
        if type(comp_forms) is not list:
            self.comp_forms = self.ncomps * [comp_forms]
        self.m_err = measurement_error

        self.components = []
        for i in range(self.ncomps):
            self.components.append(
                Component(self.pars[i], self.comp_forms[i])
            )

        if savedir is None:
            self.savedir = ''
        else:
            self.savedir = savedir.rstrip('/') + '/'
        if tablefilename is None:
            self.tablefilename = 'synthetic_data.fits'


    def extractDataAsArray(self, colnames=None, table=None):
        result = []
        if table is None:
            table = self.astr_table
        for colname in colnames:
            result.append(np.array(table[colname]))
        return np.array(result).T

    @staticmethod
    def generateSynthDataFromFile():
        """Given saved files, generate a SynthData object"""
        pass

    def generateInitXYZUVW(self, component, starcount, component_name=''):
        """Generate initial xyzuvw based on component"""
        init_size = len(self.astr_table)
        init_xyzuvw = np.random.multivariate_normal(
            mean=component.mean, cov=component.covmatrix,
            size=starcount,
        )

        # constract new table with same fields as self.astr_table,
        # then append to existing table
        names = np.arange(init_size, init_size+starcount).astype(np.str)
        new_data = Table(
            data=np.zeros(starcount,dtype=self.astr_table.dtype)
        )

        new_data['name'] = names
        new_data['component'] = starcount*[component_name]
        new_data['age'] = starcount*[component.age]
        for col, dim in zip(init_xyzuvw.T, 'xzyuvw'):
            new_data[dim+'0'] = col
        self.astr_table = vstack((self.astr_table, new_data))


    def generateAllInitXYZUVW(self):
        self.astr_table = Table(names=self.DEFAULT_NAMES,
                                dtype=self.DEFAULT_DTYPES)
        for ix, comp in enumerate(self.components):
            self.generateInitXYZUVW(comp, self.starcounts[ix],
                                    component_name=str(ix))


    def projectStars(self):
        """Project stars from xyzuvw then to xyzuvw now based on their age"""
        for star in self.astr_table:
            mean_then = self.extractDataAsArray(
                table=star,
                colnames=[dim+'0' for dim in 'xyzuvw'],
            )
            xyzuvw_now = traceorbit.traceOrbitXYZUVW(mean_then,
                                                     times=star['age'])
            for ix, dim in enumerate('xyzuvw'):
                star[dim+'_now'] = xyzuvw_now[ix]


    def measureXYZUVW(self):
        """
        Convert current day cartesian phase-space coordinates into astrometry
        values, with incorporated measurement uncertainty.
        """
        # Grab xyzuvw data in array form
        xyzuvw_now_colnames = [dim + '_now' for dim in 'xyzuvw']
        xyzuvw_now = self.extractDataAsArray(colnames=xyzuvw_now_colnames)

        # Build array of measurement errors, based on Gaia DR2 and scaled by
        # `m_err`
        # [ra, dec, plx, pmra, pmdec, rv]
        errors = self.m_err *\
                 np.array([
                     self.GERROR[colname + '_error']
                     for colname in self.DEFAULT_ASTR_COLNAMES
                 ])

        # Get perfect astrometry
        astr = coordinate.convertManyLSRXYZUVWToAstrometry(xyzuvw_now)

        # Measurement errors are applied homogenously across data so we
        # can just tile to produce uncertainty
        nstars = xyzuvw_now.shape[0]
        raw_errors = np.tile(errors, (nstars, 1))

        # Generate and apply a set of offsets from a 1D Gaussian with std
        # equal to the measurement error for each value
        offsets = raw_errors * np.random.randn(*raw_errors.shape)
        astr_w_offsets = astr + offsets

        # insert into Table
        for ix, astr_name in enumerate(self.DEFAULT_ASTR_COLNAMES):
            self.astr_table[astr_name] = astr_w_offsets[:,ix]
            self.astr_table[astr_name + '_error'] = raw_errors[:,ix]


    def storeTable(self, savedir=None, filename=None, overwrite=False):
        """
        Store table on disk.

        Parameters
        ----------
        savedir : str {None}
            the directory to store table file in
        filename : str {None}
            what to call the file (can also just use this and provide whole
            path)
        overwrite : boolean {False}
            Whether to overwrite a table in the same location
        """
        if savedir is None:
            savedir = self.savedir
        else:
            savedir = savedir.rstrip('/') + '/'
        if filename is None:
            filename = self.tablefilename
        self.astr_table.write(savedir+filename, overwrite=overwrite)


    def synthesiseEverything(self, savedir=None, filename=None, overwrite=False):
        """
        Uses self.pars and self.starcounts to generate an astropy table with
        synthetic stellar measurements.
        """
        self.generateAllInitXYZUVW()
        self.projectStars()
        self.measureXYZUVW()
        if filename is not None:
            self.storeTable(savedir=savedir, filename=filename,
                            overwrite=overwrite)