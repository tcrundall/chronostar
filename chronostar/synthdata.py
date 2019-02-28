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

#
# def synthesiseXYZUVW(pars, nstars, form='sphere', return_group=False,
#                      xyzuvw_savefile='', group_savefile='', internal=False):
#     """
#     Generate a bunch of stars in situ based off a Guassian parametrisation
#
#     Parameters
#     ----------
#     pars : [10] or [15] float array
#         10 parameters : [X,Y,Z,U,V,W,dX,dV,age,nstars]
#             Covariance matrix describes a spherical distribution in pos
#             and vel space
#         15 parameters : [X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars]
#             Covariance matrix descirbes a spherical distribution in velocity
#             space and freely orientable, triaxial ellipsoid in position space
#     nstars : int
#         Number of stars to be drawn
#     form : string {'sphere'}
#         'sphere' if providing pars in 9 parameter form,
#         'ellpitical' if providing pars in 14 parameter form,
#     return_group : boolean {False}
#         Set flag if want to return the group object (for tracking input
#         parameters)
#     xyzuvw_savefile : String {''}
#         Provide a string to numpy.save the init_xyzuvw array
#     group_savefile : Stirng {''}
#         Provide a string to numpy.save the group object; note you need to
#         np.load(group_savefile).item() in order to retrieve it.
#     internal : Boolean {False}
#         Set if parameters are provided in emcee internalised form
#
#     Returns
#     -------
#     xyzuvw_init : [nstars,6] float array
#         Initial distribution of stars in XYZUVW coordinates in corotating, RH
#         (X,U positive towards galactic anti-centre) cartesian coordinates
#         centred on local standard fo rest.
#
#     (if flag return_group is set)
#     group : SynthGroup object
#         An object that wraps initialisation parameters
#     """
#     logging.debug("Internal?: {}".format(internal))
#     group = Component(pars, form=form, internal=internal)
#     logging.debug("Mean {}".format(group.mean))
#     logging.debug("Cov\n{}".format(group.generateCovMatrix()))
#     logging.debug("Number of stars {}".format(group.nstars))
#     init_xyzuvw = np.random.multivariate_normal(
#         mean=group.mean, cov=group.generateCovMatrix(),
#         size=nstars,
#     )
#     if xyzuvw_savefile:
#         np.save(xyzuvw_savefile, init_xyzuvw)
#     if group_savefile:
#         np.save(group_savefile, group)
#     if return_group:
#         return init_xyzuvw, group
#     else:
#         return init_xyzuvw

# def synthesiseManyXYZUVW(many_pars, star_counts, form='sphere', return_groups=False,
#                          xyzuvw_savefile='', groups_savefile='',
#                          internal=False):
#     """
#     Generate a bunch of stars in situ from many Gaussian parametrisations
#
#     Note: there is no orbital projection, only the initial positions
#     of the stars are returned
#     As a consequence, this function is stupid and useless....
#
#     Parameters
#     ----------
#     many_pars : [ngroups, 10] or [ngroups, 15] float array
#         10 parameters : [X,Y,Z,U,V,W,dX,dV,age,nstars]
#             Covariance matrix describes a spherical distribution in pos
#             and vel space
#         15 parameters : [X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars]
#             Covariance matrix descirbes a spherical distribution in velocity
#             space and freely orientable, triaxial ellipsoid in position space
#     sphere : boolean {True}
#         Set flag True if providing pars in 9 parameter form,
#         Set flag False if providing pars in 14 parameter form,
#     return_groups : boolean {False}
#         Set flag if want to return the group object (for tracking input
#         parameters)
#     xyzuvw_savefile : String {''}
#         Provide a string to numpy.save the init_xyzuvw array
#     groups_savefile : Stirng {''}
#         Provide a string to numpy.save the group object; note you need to
#         np.load(group_savefile).item() in order to retrieve it.
#     internal : Boolean {False}
#         Set if parameters are provided in emcee internalised form
#
#     Returns
#     -------
#     xyzuvw_init : [Nstars,6] float array
#         Initial distribution of stars in XYZUVW coordinates in corotating, RH
#         (X,U positive towards galactic anti-centre) cartesian coordinates
#         centred on local standard fo rest.
#         Nstars is the sum of all group pars' nstars
#
#     (if flag return_group is set)
#     groups : [ngroups] Group object
#         Objects that wrap initialisation parameters
#     """
#     many_pars_cp = np.copy(many_pars)
#
#     groups = []
#     all_init_xyzuvw = np.zeros((0,6))
#
#     for pars, nstars in zip(many_pars_cp, star_counts):
#         init_xyzuvw, group = synthesiseXYZUVW(pars, nstars=nstars,
#                                               form=form,
#                                               return_group=True,
#                                               internal=internal)
#         groups.append(group)
#         all_init_xyzuvw = np.vstack((all_init_xyzuvw, init_xyzuvw))
#     if xyzuvw_savefile:
#         np.save(xyzuvw_savefile, all_init_xyzuvw)
#     if groups_savefile:
#         np.save(groups_savefile, groups)
#     if return_groups:
#         return all_init_xyzuvw, groups
#     else:
#         return all_init_xyzuvw
from chronostar.measurer import GERROR, convertArrayToTable


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
        # 'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr',
        # 'ra_pmdec_corr', 'dec_parallax_corr', 'dec_pmra_corr',
        # 'dec_pmdec_corr', 'parallax_pmra_corr', 'parallax_pmdec_corr',
        # 'pmra_pmdec_corr',
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


    def measureStars(self):
        astr_w_errs = None


    def generateComponentData(self):
        """Generate the full astrometry for a component"""
        # initialise blank astropy astrometry table
        self.generateAllInitXYZUVW()
        # self.projectStars()


    def measureXYZUVW(self):
        """
        TODO: Work out neat way to measure stars in place.
        """
        xyzuvw_now_colnames = [dim + '_now' for dim in 'xyzuvw']
        xyzuvw_now = self.extractDataAsArray(colnames=xyzuvw_now_colnames)

        errors = np.array([
            self.GERROR[colname + '_error']
            for colname in self.DEFAULT_ASTR_COLNAMES
        ])

        nstars = xyzuvw_now.shape[0]
        astr = coordinate.convertManyLSRXYZUVWToAstrometry(xyzuvw_now)

        raw_errors = self.m_err * np.tile(errors, (nstars, 1))
        offsets = raw_errors * np.random.randn(*raw_errors.shape)
        astr_w_offsets = astr + offsets

        # insert into Table
        for ix, astr_name in enumerate(self.DEFAULT_ASTR_COLNAMES):
            self.astr_table[astr_name] = astr_w_offsets[:,ix]
            self.astr_table[astr_name + '_error'] = raw_errors[:,ix]


    def storeTable(self, savedir=None, filename=None, overwrite=False):
        if savedir is None:
            savedir = self.savedir
        else:
            savedir = savedir.rstrip('/') + '/'
        if filename is None:
            filename = self.tablefilename
        self.astr_table.write(savedir+filename, overwrite=overwrite)


    def synthesiseEverything(self, savedir=None, filename=None, overwrite=True):
        self.generateAllInitXYZUVW()
        self.projectStars()
        self.measureXYZUVW()
        self.storeTable(savedir=savedir, filename=filename,
                        overwrite=overwrite)


def redundantDocString():
    """
    Returns dictionary of measurments, so that this method can be
    used without an internal table.

    Replicates the measurement of synthetic stars. Converts XYZUVW to radec..

    Parameters
    ----------
    xyzuvws : [nstars, 6] float array
        A list of stars in rh cartesian
        coordinate system, centred on and co-rotating with the local standard
        of rest
        [pc, pc, pc, km/s, km/s, km/s]
    error_frac : float
        Parametrisation of Gaia-like uncertainty. 0 is perfect precision,
        1.0 is simplified best Gaia uncertainty.
    savefile : string {''}
        if not empty, the astrometry table will be saved to the given
        file name

    Returns
    -------
    real_astros : ([nstars, 6] float array) List of stars in measurements with
        incorporated error
    """
    pass
