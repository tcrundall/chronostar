"""
Class object that encapsulates a component, the phase-space model
of an unbound set of stars formed from the same starburst/filament.

A component models the initial phase-space distribution of stars
as a Gaussian. As such there are three key attributes:
- mean: the central location
- covariance matrix: the spread in each dimension along with any correlations
- age: how long the stars have been travelling
"""

from __future__ import print_function, division, unicode_literals

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats.mstats import gmean

from . import transform
from chronostar.traceorbit import traceOrbitXYZUVW
from . import traceorbit


class AbstractComponent(object):
    __metaclass__ = ABCMeta

    DEFAULT_TINY_AGE = 1e-10

    _pars = None
    _mean = None
    _covmatrix = None
    _age = None
    _sphere_dx = None
    _sphere_dv = None

    _mean_now = None
    _covmatrix_now = None

    # Set these in concrete class, matching form with 'SENSIBLE_WALKER_SPREADS'
    # See SphereComponent and EllipComponent for examples
    PARAMETER_FORMAT = None

    SENSIBLE_WALKER_SPREADS = {
        'pos':10.,
        'pos_std':1.,
        'log_pos_std':0.5,
        'vel':2.,
        'vel_std':1.,
        'log_vel_std':0.5,
        'corr':0.05,
        'age':1.,
    }

    def __init__(self, pars=None, attributes={}, internal=False,
                 trace_orbit_func=traceOrbitXYZUVW):
        # Some basic implementation checks
        self.check_parameter_format()

        # Set cartesian orbit tracing function
        self.trace_orbit_func = trace_orbit_func

        # If parameters are provided in internal form (the form used by emcee),
        # then externalise before setting of various other attributes.
        if pars is not None:
            if internal:
                self._pars = self.externalise(pars)
            else:
                self._pars = np.copy(pars)
        else:
            self._pars = np.zeros(len(self.PARAMETER_FORMAT))

            # Age *must* be non-zero
            self._set_age(self.DEFAULT_TINY_AGE)

        # Using provided parameters, set up the three model attributes:
        # mean, covariance and age. If attributes are provided, then use
        # those.
        self._set_mean(attributes.get('mean', None))
        self._set_covmatrix(attributes.get('covmatrix', None))
        self._set_age(attributes.get('age', None))

        # For some purposes (e.g. virialisation estimation) it is useful to
        # approximate position and velocity volumes as spherical. Calculate
        # and set those attributes.
        self.set_sphere_stds()

    @classmethod
    def check_parameter_format(cls):
        if cls.PARAMETER_FORMAT is None:
            raise NotImplementedError('Need to define PARAMETER_FORMAT '
                                      'as a class parameter')
        if not np.all(np.isin(cls.PARAMETER_FORMAT,
                              list(cls.SENSIBLE_WALKER_SPREADS.keys()))):
            raise NotImplementedError('Label in PARAMETER_FORMAT doesn\'t '
                                      'seem to be in SENSIBLE_WALKER_SPREADS. '
                                      'Extend dictionary in AbstractComponent '
                                      'accordingly')

    @classmethod
    def externalise(cls, pars):
        raise NotImplementedError

    @classmethod
    def internalise(cls, pars):
        raise NotImplementedError

    def get_pars(self):
        return np.copy(self._pars)

    @abstractmethod
    def _set_mean(self, mean=None): pass

    def get_mean(self):
        return np.copy(self._mean)

    @abstractmethod
    def _set_covmatrix(self, covmatrix=None): pass

    def get_covmatrix(self):
        return np.copy(self._covmatrix)

    @abstractmethod
    def _set_age(self, age=None): pass

    def get_age(self):
        return self._age

    def get_attributes(self):
        return {'mean':self.get_mean(),
                'covmatrix':self.get_covmatrix(),
                'age':self.get_age()}

    def set_sphere_stds(self):
        self._sphere_dx = gmean(np.sqrt(
                np.linalg.eigvalsh(self._covmatrix[:3, :3]))
        )
        self._sphere_dv = gmean(np.sqrt(
                np.linalg.eigvalsh(self._covmatrix[3:, 3:]))
        )

    def get_sphere_dx(self):
        if self._sphere_dx is None:
            self.set_sphere_stds()
        return self._sphere_dx

    def get_sphere_dv(self):
        if self._sphere_dv is None:
            self.set_sphere_stds()
        return self._sphere_dv

    def update_attribute(self, attributes={}):
        """
        A potential source of subtle bugs is that one can modify attributes
        (say mean) but if `covmatrix_now` has already been calculated, it
        won't update. So ideally use this method to modify attributes so
        we can force the recalculation of current-day projections as required
        """
        if 'mean' in attributes.keys():
            self._set_mean(mean=attributes['mean'])
        if 'covmatrix' in attributes.keys():
            self._set_covmatrix(covmatrix=attributes['covmatrix'])
        if 'age' in attributes.keys():
            self._set_age(age=attributes['age'])
        self._mean_now = None
        self._covmatrix_now = None


    def get_mean_now(self):
        """
        Calculates the mean of the component when projected to the current-day
        """
        if self._mean_now is None:
            self._mean_now =\
                self.trace_orbit_func(self._mean, times=self._age)
        return self._mean_now

    def get_covmatrix_now(self):
        """
        Calculates covariance matrix of current day distribution.

        Calculated as a first-order Taylor approximation of the coordinate
        transformation that takes the initial mean to the current day mean.
        This is the most expensive aspect of Chronostar, so we first make
        sure the covariance matrix hasn't already been projected.
        """
        if self._covmatrix_now is None:
            self._covmatrix_now = transform.transformCovMat(
                    self._covmatrix, trans_func=self.trace_orbit_func,
                    loc=self._mean, args=(self._age,),
            )
        return self._covmatrix_now

    def get_currentday_projection(self):
        """
        Calculate (as needed) and return the current day projection of Component

        Returns
        -------
        mean_now : [6] float array_like
            The phase-space centroid of current-day Gaussian distribution of
            Component
        covmatrix_now : [6,6] float array_like
            The phase-space covariance matrix of current-day Gaussian
            distribution of Component
        """
        return self.get_mean_now(), self.get_covmatrix_now()

    def splitGroup(self, lo_age, hi_age):
        """
        Generate two new components that share the current day mean, and
        initial covariance matrix of this component but with different ages:
        `lo_age` and `hi_age`.

        Parameters
        ----------
        lo_age : float
            Must be a positive (and ideally smaller) value than self.age.
            Serves as the age for the younger component.
        hi_age : float
            Must be a positive (and ideally larger) value than self.age
            Serves as the age for the older component.

        Returns
        -------
        lo_comp : Component
            A component that matches `self` in current-day mean and initial
            covariance matrix but with a younger age
        hi_comp : Component
            A component that matches `self` in current-day mean and initial
            covariance matrix but with an older age
        """
        comps = []
        for new_age in [lo_age, hi_age]:
            # Give new component identical initial covmatrix, and a initial
            # mean chosen to yield identical mean_now
            new_mean = self.trace_orbit_func(self.get_mean_now(),
                                             times=-new_age)
            new_comp = self.__class__(attributes={'mean':new_mean,
                                                  'covmatrix':self._covmatrix,
                                                  'age':new_age})
            comps.append(new_comp)

        return comps

    @staticmethod
    def load_components(filename):
        """
        Load Component objects from a *.npy file.

        Used to standardise result if loading a single component vs multiple
        components.

        Parameters
        ----------
        filename : str
            name of the stored file

        Returns
        -------
        res : [Component] list
            A list of Component objects
        """
        res = np.load(filename)
        if res.shape == ():
            return [res.item()]
        else:
            return res

    @classmethod
    def get_sensible_walker_spread(cls):
        sensible_spread = []
        for par_form in cls.PARAMETER_FORMAT:
            sensible_spread.append(cls.SENSIBLE_WALKER_SPREADS[par_form])
        return np.array(sensible_spread)


class SphereComponent(AbstractComponent):
    PARAMETER_FORMAT = ['pos', 'pos', 'pos', 'vel', 'vel', 'vel',
                        'log_pos_std', 'log_vel_std',
                        'age']

    @classmethod
    def externalise(cls, pars):
        extern_pars = np.copy(pars)
        extern_pars[6:8] = np.exp(extern_pars[6:8])
        return extern_pars

    @classmethod
    def internalise(cls, pars):
        intern_pars = np.copy(pars)
        intern_pars[6:8] = np.log(intern_pars[6:8])
        return intern_pars

    def _set_mean(self, mean=None):
        """Builds mean from self.pars. If setting from an externally
        provided mean then updates self.pars for consistency"""
        if mean is None:
            self._mean = self._pars[:6]
        else:
            self._mean = np.copy(mean)
            self._pars[:6] = self._mean

    def _set_covmatrix(self, covmatrix=None):
        """Builds covmatrix from self.pars. If setting from an externally
        provided covariance matrix then updates self.pars for consistency"""
        if covmatrix is None:
            dx = self._pars[6]
            dv = self._pars[7]
            self._covmatrix = np.identity(6)
            self._covmatrix[:3, :3] *= dx ** 2
            self._covmatrix[3:, 3:] *= dv ** 2
        else:
            self._covmatrix = np.copy(covmatrix)
            dx = gmean(np.sqrt(
                np.linalg.eigvalsh(self._covmatrix[:3, :3]))
            )
            dv = gmean(np.sqrt(
                np.linalg.eigvalsh(self._covmatrix[3:, 3:]))
            )
            self._pars[6] = dx
            self._pars[7] = dv
            self.set_sphere_stds()

    def _set_age(self, age=None):
        """Builds age from self.pars. If setting from an externally
        provided age then updates self.pars for consistency"""
        if age is None:
            self._age = self._pars[-1]
        else:
            self._age = age
            self._pars[-1] = age


class EllipComponent(AbstractComponent):
    PARAMETER_FORMAT = ['pos', 'pos', 'pos', 'vel', 'vel', 'vel',
                        'log_pos_std', 'log_pos_std', 'log_pos_std',
                        'log_vel_std',
                        'corr', 'corr', 'corr',
                        'age']

    @classmethod
    def externalise(cls, pars):
        extern_pars = np.copy(pars)
        extern_pars[6:10] = np.exp(extern_pars[6:10])
        return extern_pars

    @classmethod
    def internalise(cls, pars):
        intern_pars = np.copy(pars)
        intern_pars[6:10] = np.log(intern_pars[6:10])
        return intern_pars

    def _set_mean(self, mean=None):
        """Builds mean from self.pars. If setting from an externally
        provided mean then updates self.pars for consistency"""
        if mean is None:
            self._mean = self._pars[:6]
        else:
            self._mean = np.copy(mean)
            self._pars[:6] = self._mean

    def _set_covmatrix(self, covmatrix=None):
        """Builds covmatrix from self.pars. If setting from an externally
        provided covariance matrix then updates self.pars for consistency"""
        if covmatrix is None:
            dx, dy, dz = self._pars[6:9]
            dv = self._pars[9]
            c_xy, c_xz, c_yz = self._pars[10:13]
            self._covmatrix = np.array([
                [dx**2,      c_xy*dx*dy, c_xz*dx*dz, 0.,    0.,    0.],
                [c_xy*dx*dy, dy**2,      c_yz*dy*dz, 0.,    0.,    0.],
                [c_xz*dx*dz, c_yz*dy*dz, dz**2,      0.,    0.,    0.],
                [0.,         0.,         0.,         dv**2, 0.,    0.],
                [0.,         0.,         0.,         0.,    dv**2, 0.],
                [0.,         0.,         0.,         0.,    0.,    dv**2],
            ])
        else:
            self._covmatrix = np.copy(covmatrix)
            pos_stds = np.sqrt(np.diagonal(self._covmatrix[:3, :3]))
            dx, dy, dz = pos_stds
            pos_corr_matrix = (self._covmatrix[:3, :3]
                               / pos_stds
                               / pos_stds.reshape(1,3).T)
            c_xy, c_xz, c_yz = pos_corr_matrix[np.triu_indices(3,1)]
            dv = gmean(np.sqrt(
                np.linalg.eigvalsh(self._covmatrix[3:, 3:]))
            )
            self._pars[6:9] = dx, dy, dz
            self._pars[9] = dv
            self._pars[10:13] = c_xy, c_xz, c_yz

    def _set_age(self, age=None):
        """Builds age from self.pars. If setting from an externally
        provided age then updates self.pars for consistency"""
        if age is None:
            self._age = self._pars[-1]
        else:
            self._age = age
            self._pars[-1] = age
