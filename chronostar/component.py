"""
Class object that encapsulates a component, the phase-space model
of an unbound set of stars formed from the same starburst/filament.

A component models the initial phase-space distribution of stars
as a Gaussian. As such there are three key attributes:
- mean: the central location
- covariance matrix: the spread in each dimension along with any correlations
- age: how long the stars have been travelling

TODO: Have actual names for parameters for clarity when logging results
"""

from __future__ import print_function, division, unicode_literals

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats.mstats import gmean

from . import transform
from chronostar.traceorbit import traceOrbitXYZUVW
from . import traceorbit


class AbstractComponent(object):
    """
    An abstract class that (when implmented) encapsulates a component,
    the phase-space model of an unbound set of stars formed from the
    same starburst/filament.

    A component models the initial phase-space distribution of stars
    as a Gaussian. As such there are three key attributes:
    - mean: the central location
    - covariance matrix: the spread in each dimension along with any
    correlations
    - age: how long the stars have been travelling

    This class has been left abstract so as to easily facilitate
    (and encourage) alternate parameterisations of components.

    In brief, just copy-paste SphereComponent below to make your new
    class, and modify the methods and attribute to suit your new
    parametrisation.

    In order to implement this class and make a concrete class, only
    one variable must be set, and four methods implmented. In short,
    the class must be told how to turn raw parameters into attributes,
    and vice verse.

    Attributes to set:
    `PARAMETER_FORMAT`
        This parameter must be set. An ordered list of labels
        describing what purpose each input serves. e.g. for a
        SphereComponent, the list is
        3*['pos'] + 3*['vel'] + ['log_pos_std', 'log_vel_std', 'age']
        See `SENSIBLE_WALKER_SPREADS` for a set of viable labels, and
        include your own as needed! Note that this is the parameters in
        "internal form", i.e. the form of the parameter space that
        emcee explores.

    Methods to define
    internalise(pars) and externalise(pars)
        You must tell the Component class how to switch between internal
        and external formats. These methods are static because there is
        not always a need to instantiate an entire Component object
        simply to convert between parameter forms.

        There is perhaps scope to have the Component class to intuit
        how to convert between forms based on `PARAMETER_FORMAT` values.
    _set_covmatrix(covmatrix=None), (REQUIRED)
    _set_mean(mean=None), _set_age(age=None) (both optional)
        These methods instruct the class how to (if input is None) build
        the attribute from self.pars, or (if input is provided) to set
        the self._mean (for e.g.) attribute but also to reverse engineer
        the self.pars values and update accordingly.

        These methods should only be called internally, (from the
        __init__() method, or the update_attributes() method) as it is
        critical to do some tidying up (setting mean_now and
        covmatrix_now to None) whenever self.pars is modified.

        If you stick to the convention of the mean=pars[:6] and
        age=pars[-1] then the default methods will suffice and you will
        only need to implement _set_covmatrix(). Of course if you wish,
        you can override _set_mean() or _set_age().
    """
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

    # This is used to guide the scale of variations in each parameter
    # Super useful when initialising emcee walkers across a sensible
    # volume of parameter space.
    SENSIBLE_WALKER_SPREADS = {
        'pos':10.,
        'pos_std':1.,
        'log_pos_std':0.5,
        'vel':2.,
        'vel_std':1.,
        'log_vel_std':0.5,
        'corr':0.05,
        'age':1.,
        'angle_rad':0.25*np.pi,
        'angle_deg':45.,
    }

    def __init__(self, pars=None, attributes=None, internal=False,
                 trace_orbit_func=None):
        """
        An abstraction for the parametrisation of a moving group
        component origin. As a 6D Gaussian, a Component has three key
        attributes; mean, covariance matrix, and age. There are many
        ways to parameterise a covariance matrix to various degrees
        of freedom.

        Parameters
        ----------
        pars: 1D float array_like
            Raw values for the parameters of the component. Can be
            provided in "external" form (standard) or "internal" form
            (e.g. treating standard deviations in log space to ensure
            uninformed prior)
        attributes: dict with all the following keys:
            mean: [6] float array_like
                The mean of the initial Gaussian distribution in
                cartesian space:
                [X(pc), Y(pc), Z(pc), U(km/s), V(km/s), W(km/s)]
            covmatrix: [6,6] float array_like
                the covariance matrix of the initial Gaussian
                distribution, with same units as `mean`
            age: float
                the age of the component (positive) in millions of
                years
        internal: boolean {False}
            If set, and if `pars` is provided, treats input pars as
            internal form, and first externalises them before building
            attributes.
        trace_orbit_func: function {traceOrbitXYZUVW}
            Function used to calculate an orbit through cartesian space
            (centred on, and co-rotating with, the local standard of
            rest). Function must be able to take two parameters, the
            starting location and the age, with positive age
            corrsponding to forward evolution, and negative age
            backward evolution. It should also be "odd", i.e.:
            func(loc_then, +age) = loc_now
            func(loc_now,  -age) = loc_then

        Returns
        -------
        res: Component object
            An astraction of a set of values parametrising the origin of
            a moving group component.
        """
        # Some basic implementation checks
        self.check_parameter_format()

        # Set cartesian orbit tracing function
        if trace_orbit_func is None:
            self.trace_orbit_func = traceOrbitXYZUVW
        else:
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
        if attributes is None:
            attributes = {}
        self._set_mean(attributes.get('mean', None))
        self._set_covmatrix(attributes.get('covmatrix', None))
        self._set_age(attributes.get('age', None))

        # For some purposes (e.g. virialisation estimation) it is useful to
        # approximate position and velocity volumes as spherical. Calculate
        # and set those attributes.
        self.set_sphere_stds()

    def __str__(self):
        x,y,z,u,v,w = self.get_mean_now()
        return 'Currentday(' \
               'X: {:.2}pc, Y: {:.2}pc, Z: {:.2}pc, '  \
               'U {:.2}km/s, V {:.2}km/s, W {:.2}km/s, ' \
               'age: {:.2}Myr)'.format(x,y,z,u,v,w, self._age)

    def __repr__(self):
        return self.__str__()

    @classmethod
    def check_parameter_format(cls):
        """
        A check for valid implementation. If this throws an error then
        the PARAMETER_FORMAT attribute has been incorrectly defined.
        """
        if cls.PARAMETER_FORMAT is None:
            raise NotImplementedError('Need to define PARAMETER_FORMAT '
                                      'as a class parameter')
        if not np.all(np.isin(cls.PARAMETER_FORMAT,
                              list(cls.SENSIBLE_WALKER_SPREADS.keys()))):
            raise NotImplementedError('Label in PARAMETER_FORMAT doesn\'t '
                                      'seem to be in SENSIBLE_WALKER_SPREADS. '
                                      'Extend dictionary in AbstractComponent '
                                      'accordingly: {}'.format(
                                              cls.PARAMETER_FORMAT
                                    ))

    @staticmethod
    def externalise(pars):
        """
        Take parameter set in internal form (as used by emcee) and
        convert to external form (as used to build attributes).

        Tasks
        -----
        There is scope to implement this here, and use cls.PARAMETER_FORMAT
        to guide the parameter conversions
        """
        raise NotImplementedError

    @staticmethod
    def internalise(pars):
        """
        Take parameter set in external form (as used to build attributes)
        and convert to internal form (as used by emcee).

        Tasks
        -----
        There is scope to implement this here, and use cls.PARAMETER_FORMAT
        to guide the parameter conversions
        """
        raise NotImplementedError

    def get_pars(self):
        """
        Return a copy of the raw (external) parameterisation of
        the Component
        """
        return np.copy(self._pars)

    def _set_mean(self, mean=None):
        """
        Builds mean from self.pars. If setting from an externally
        provided mean then updates self.pars for consistency

        If implementation does use the first 6 values in self._pars
        to set the mean then this method should be overridden.
        """
        # If mean hasn't been provided, generate from self._pars
        # and set.
        if mean is None:
            self._mean = self._pars[:6]
        # If mean has been provided, reverse engineer and update
        # self._pars accordingly.
        else:
            self._mean = np.copy(mean)
            self._pars[:6] = self._mean

    def get_mean(self):
        """Return a copy of the mean (initial) of the component"""
        return np.copy(self._mean)

    @abstractmethod
    def _set_covmatrix(self, covmatrix=None):
        """
        Builds covmatrix from self._pars. If setting from an externally
        provided covmatrix then update self._pars for consistency.

        This is the sole method that needs implmentation to build a
        usable Component class
        """
        pass

    def get_covmatrix(self):
        """Return a copy of the covariance matrix (initial)"""
        return np.copy(self._covmatrix)

    def _set_age(self, age=None):
        """Builds age from self.pars. If setting from an externally
        provided age then updates self.pars for consistency"""
        if age is None:
            self._age = self._pars[-1]
        else:
            self._age = age
            self._pars[-1] = age

    def get_age(self):
        """Returns the age of the Component"""
        return self._age

    def get_attributes(self):
        """
        Get a dictionary of all three key attributes of the Component
        model. Done this way for easy of initialising a new Component.
        """
        return {'mean':self.get_mean(),
                'covmatrix':self.get_covmatrix(),
                'age':self.get_age()}

    def set_sphere_stds(self):
        """
        Set the spherical standard deviations in position space and
        velocity space. Calculated in such a way so as to preserved
        volume in position space and velocity space retrospectively.
        Note that combined phase-space volume is not conserved by this
        implementation.
        """
        self._sphere_dx = gmean(np.sqrt(
                np.linalg.eigvalsh(self._covmatrix[:3, :3]))
        )
        self._sphere_dv = gmean(np.sqrt(
                np.linalg.eigvalsh(self._covmatrix[3:, 3:]))
        )

    def get_sphere_dx(self):
        """
        Return the spherical standard deviation in position space.

        First check if it is None (which may be the case if covmatrix
        has been updated for e.g.) and recalculate at need.
        """
        if self._sphere_dx is None:
            self.set_sphere_stds()
        return self._sphere_dx

    def get_sphere_dv(self):
        """
        Return the spherical standard deviation in velocity space.

        First check if it is None (which may be the case if covmatrix
        has been updated for e.g.) and recalculate at need.
        """
        if self._sphere_dv is None:
            self.set_sphere_stds()
        return self._sphere_dv

    def update_attribute(self, attributes=None):
        """
        Update attributes based on input dictionary.

        Parameters
        ----------
        attributes: dict
            A dictionary with the any combination (including none) of the
            following:
            'mean': [6] float array_like
                the mean of the initial 6D Gaussian
            'covmatrix': [6,6] float array_like
                the covariance matrix of the initial 6D Gaussian
            'age': float
                the age of the component

        Notes
        -----
        A potential source of subtle bugs is that one can modify attributes
        (e.g. mean) but if `covmatrix_now` has already been calculated, it
        won't update. So it is critical to use only this method to modify
        attributes such that we can force the recalculation of current-day
        projections as required.
        """
        if type(attributes) is not dict:
            raise TypeError('Attributes must be passed in as dictionary')
        if 'mean' in attributes.keys():
            self._set_mean(mean=attributes['mean'])
        if 'covmatrix' in attributes.keys():
            self._set_covmatrix(covmatrix=attributes['covmatrix'])
        if 'age' in attributes.keys():
            self._set_age(age=attributes['age'])
        self._mean_now = None
        self._covmatrix_now = None
        self._sphere_dx = None
        self._sphere_dv = None

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
            return np.array([res.item()])
        else:
            return res

    @classmethod
    def get_sensible_walker_spread(cls):
        """Get an array of sensible walker spreads (based on class
        constants `PARAMTER_FORMAT` and `SENSIBLE_WALKER_SPREADS` to
        guide emcee in a sensible starting range of parameters."""
        sensible_spread = []
        for par_form in cls.PARAMETER_FORMAT:
            sensible_spread.append(cls.SENSIBLE_WALKER_SPREADS[par_form])
        return np.array(sensible_spread)


class SphereComponent(AbstractComponent):
    PARAMETER_FORMAT = ['pos', 'pos', 'pos', 'vel', 'vel', 'vel',
                        'log_pos_std', 'log_vel_std',
                        'age']

    @staticmethod
    def externalise(pars):
        """
        Take parameter set in internal form (as used by emcee) and
        convert to external form (as used to build attributes).
        """
        extern_pars = np.copy(pars)
        extern_pars[6:8] = np.exp(extern_pars[6:8])
        return extern_pars

    @staticmethod
    def internalise(pars):
        """
        Take parameter set in external form (as used to build attributes)
        and convert to internal form (as used by emcee).
        """
        intern_pars = np.copy(pars)
        intern_pars[6:8] = np.log(intern_pars[6:8])
        return intern_pars

    def _set_covmatrix(self, covmatrix=None):
        """Builds covmatrix from self.pars. If setting from an externally
        provided covariance matrix then updates self.pars for consistency"""
        # If covmatrix hasn't been provided, generate from self._pars
        # and set.
        if covmatrix is None:
            dx = self._pars[6]
            dv = self._pars[7]
            self._covmatrix = np.identity(6)
            self._covmatrix[:3, :3] *= dx ** 2
            self._covmatrix[3:, 3:] *= dv ** 2
        # If covmatrix has been provided, reverse engineer the most
        # suitable set of parameters and update self._pars accordingly
        # (e.g. take the geometric mean of the (square-rooted) velocity
        # eigenvalues as dv, as this at least ensures constant volume
        # in velocity space).
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


class EllipComponent(AbstractComponent):
    PARAMETER_FORMAT = ['pos', 'pos', 'pos', 'vel', 'vel', 'vel',
                        'log_pos_std', 'log_pos_std', 'log_pos_std',
                        'log_vel_std',
                        'corr', 'corr', 'corr',
                        'age']

    @staticmethod
    def externalise(pars):
        """
        Take parameter set in internal form (as used by emcee) and
        convert to external form (as used to build attributes).
        """
        extern_pars = np.copy(pars)
        extern_pars[6:10] = np.exp(extern_pars[6:10])
        return extern_pars

    @staticmethod
    def internalise(pars):
        """
        Take parameter set in external form (as used to build attributes)
        and convert to internal form (as used by emcee).
        """
        intern_pars = np.copy(pars)
        intern_pars[6:10] = np.log(intern_pars[6:10])
        return intern_pars

    def _set_covmatrix(self, covmatrix=None):
        """Builds covmatrix from self.pars. If setting from an externally
        provided covariance matrix then updates self.pars for consistency"""
        # If covmatrix hasn't been provided, generate from self._pars
        # and set.
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
        # If covmatrix has been provided, reverse engineer the most
        # suitable set of parameters and update self._pars accordingly
        # (e.g. take the geometric mean of the (square-rooted) velocity
        # eigenvalues as dv, as this at least ensures constant volume
        # in velocity space).
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


