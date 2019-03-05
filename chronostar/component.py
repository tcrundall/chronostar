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
import logging
import numpy as np
from scipy.stats.mstats import gmean

from . import transform
from . import traceorbit


class AbstractComponent(object):
    __metaclass__ = ABCMeta

    DEFAULT_TINY_AGE = 1e-10

    pars = None
    mean = None
    covmatrix = None
    age = None
    sphere_dx = None
    sphere_dv = None

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

    def __init__(self, pars=None, attributes={}, internal=False):
        # Some basic implementation checks
        self.check_parameter_format()

        # If parameters are provided in internal form (the form used by emcee),
        # then externalise before setting of various other attributes.
        if pars is not None:
            if internal:
                self.pars = self.externalise(pars)
            else:
                self.pars = np.copy(pars)
        else:
            self.pars = np.zeros(len(self.PARAMETER_FORMAT))

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
        return np.copy(self.pars)

    @abstractmethod
    def _set_mean(self, mean=None): pass

    def get_mean(self):
        return np.copy(self.mean)

    @abstractmethod
    def _set_covmatrix(self, covmatrix=None): pass

    def get_covmatrix(self):
        return np.copy(self.covmatrix)

    @abstractmethod
    def _set_age(self, age=None): pass

    def get_age(self):
        return self.age

    def get_attributes(self):
        return {'mean':self.get_mean(),
                'covmatrix':self.get_covmatrix(),
                'age':self.get_age()}

    def set_sphere_stds(self):
        self.sphere_dx = gmean(np.sqrt(
                np.linalg.eigvalsh(self.covmatrix[:3, :3]))
        )
        self.sphere_dv = gmean(np.sqrt(
                np.linalg.eigvalsh(self.covmatrix[3:, 3:]))
        )

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
                traceorbit.traceOrbitXYZUVW(self.mean, times=self.age)
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
                    self.covmatrix, trans_func=traceorbit.traceOrbitXYZUVW,
                    loc=self.mean, args=(self.age,),
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
            covariance matrix but wiht an older age
        """
        if self._mean_now is None:
            self.get_mean_now()

        comps = []
        for age in [lo_age, hi_age]:
            # Give new component identical initial covmatrix, and a initial
            # mean chosen to yield identical mean_now
            new_mean = traceorbit.traceOrbitXYZUVW(self._mean_now, times=-age)
            new_comp = self.__class__(attributes={'mean':new_mean,
                                                  'covmatrix':self.covmatrix,
                                                  'age':age})
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
            self.mean = self.pars[:6]
        else:
            self.mean = np.copy(mean)
            self.pars[:6] = self.mean

    def _set_covmatrix(self, covmatrix=None):
        """Builds covmatrix from self.pars. If setting from an externally
        provided covariance matrix then updates self.pars for consistency"""
        if covmatrix is None:
            dx = self.pars[6]
            dv = self.pars[7]
            self.covmatrix = np.identity(6)
            self.covmatrix[:3,:3] *= dx**2
            self.covmatrix[3:,3:] *= dv**2
        else:
            self.covmatrix = np.copy(covmatrix)
            dx = gmean(np.sqrt(
                np.linalg.eigvalsh(self.covmatrix[:3, :3]))
            )
            dv = gmean(np.sqrt(
                np.linalg.eigvalsh(self.covmatrix[3:, 3:]))
            )
            self.pars[6] = dx
            self.pars[7] = dv
            self.set_sphere_stds()

    def _set_age(self, age=None):
        """Builds age from self.pars. If setting from an externally
        provided age then updates self.pars for consistency"""
        if age is None:
            self.age = self.pars[-1]
        else:
            self.age = age
            self.pars[-1] = age


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
            self.mean = self.pars[:6]
        else:
            self.mean = np.copy(mean)
            self.pars[:6] = self.mean

    def _set_covmatrix(self, covmatrix=None):
        """Builds covmatrix from self.pars. If setting from an externally
        provided covariance matrix then updates self.pars for consistency"""
        if covmatrix is None:
            dx, dy, dz = self.pars[6:9]
            dv = self.pars[9]
            c_xy, c_xz, c_yz = self.pars[10:13]
            self.covmatrix = np.array([
                [dx**2,      c_xy*dx*dy, c_xz*dx*dz, 0.,    0.,    0.],
                [c_xy*dx*dy, dy**2,      c_yz*dy*dz, 0.,    0.,    0.],
                [c_xz*dx*dz, c_yz*dy*dz, dz**2,      0.,    0.,    0.],
                [0.,         0.,         0.,         dv**2, 0.,    0.],
                [0.,         0.,         0.,         0.,    dv**2, 0.],
                [0.,         0.,         0.,         0.,    0.,    dv**2],
            ])
        else:
            self.covmatrix = np.copy(covmatrix)
            pos_stds = np.sqrt(np.diagonal(self.covmatrix[:3,:3]))
            dx, dy, dz = pos_stds
            pos_corr_matrix = (self.covmatrix[:3,:3]
                               / pos_stds
                               / pos_stds.reshape(1,3).T)
            c_xy, c_xz, c_yz = pos_corr_matrix[np.triu_indices(3,1)]
            dv = gmean(np.sqrt(
                np.linalg.eigvalsh(self.covmatrix[3:, 3:]))
            )
            self.pars[6:9] = dx, dy, dz
            self.pars[9] = dv
            self.pars[10:13] = c_xy, c_xz, c_yz

    def _set_age(self, age=None):
        """Builds age from self.pars. If setting from an externally
        provided age then updates self.pars for consistency"""
        if age is None:
            self.age = self.pars[-1]
        else:
            self.age = age
            self.pars[-1] = age

class Component(object):
    IMPLEMENTED_FORMS = ('sphere', 'elliptical')
    DEFAULT_TINY_AGE = 1e-5

    mean_now = None
    covmatrix_now = None

    sphere_dx = None
    sphere_dv = None

    @staticmethod
    def getSensibleInitSpread(form='sphere'):
        pos_spread = 10.
        vel_spread =  2.
        log_pos_stdev_spread = 0.5
        log_vel_stdev_spread = 0.5
        pos_corr_spread = 0.05
        age_spread = 1.
        if form == 'sphere':
            #                 X,  Y,  Z,U, V, W, lndX,lndV,age
            return np.array([
                pos_spread, pos_spread, pos_spread,
                vel_spread, vel_spread, vel_spread,
                log_pos_stdev_spread, log_vel_stdev_spread, age_spread
            ])
        elif form == 'elliptical':
            #                 X,  Y,  Z,U, V, W, lndX,lndV,age
            return np.array([
                pos_spread, pos_spread, pos_spread,
                vel_spread, vel_spread, vel_spread,
                log_pos_stdev_spread, log_pos_stdev_spread,
                log_pos_stdev_spread, log_vel_stdev_spread,
                pos_corr_spread, pos_corr_spread, pos_corr_spread,
                age_spread,
            ])


    @staticmethod
    def loadComponents(filename):
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

    @staticmethod
    def externalisePars(pars, form='sphere'):
        """
        Convert parameters from internal form to external form.

        Parameters
        ----------
        pars : [n] float array_like
            Raw float parametrisation of a component in internal form.
            Main differences between internal and external is that standard
            deviations are stored in log form.
            See signature for __init__ for detailed breakdown of format of
            `pars`
        form : string {'sphere'}
            Possible values ['sphere'|'elliptical']
            Determines the arrangement of values in `pars`

        Returns
        -------
        pars : [n] float array_like
            Pars in external format
        """
        pars = np.copy(pars)
        if form == 'sphere':
            pars[6:8] = np.exp(pars[6:8])
        elif form == 'elliptical':
            pars[6:10] = np.exp(pars[6:10])
        return pars

    @staticmethod
    def internalisePars(pars, form='sphere'):
        """
        Convert parameters from internal form to external form.

        Parameters
        ----------
        pars : [n] float array_like
            Raw float parametrisation of a component in internal form.
            Main differences between internal and external is that standard
            deviations are stored in log form.
            See signature for __init__ for detailed breakdown of format of
            `pars`
        form : string {'sphere'}
            Possible values ['sphere'|'elliptical']
            Determines the arrangement of values in `pars`

        Returns
        -------
        pars : [n] float array_like
            Pars in external format
        """
        pars = np.copy(pars)
        if form == 'sphere':
            pars[6:8] = np.log(pars[6:8])
        elif form == 'elliptical':
            pars[6:10] = np.log(pars[6:10])
        return pars

    def __init__(self, pars=None, form='sphere', internal=False,
                 mean=None, covmatrix=None):
        """
        Parameters
        ----------
        pars : [n] float array_like
            The raw parameterisation of the component's origin and age
        form : string ['sphere'|'elliptical'|'filament']
            Determines how to parse the various parameterisations.
            spherical [9]: [x,y,z,u,v,w,dx,dv,age]
                Spherical (isotropic) in both position and velocity with no
                correlation between any axes.
            elliptical [14]: [x,y,z,u,v,w,dx,dy,dz,dv,cxy,cxz,cyz,age]
                Freely rotatable in position space but spherical (isotropic)
                in velocity space. Correlations only exist between position
                axes.
            filament [15]: [x,y,z,u,v,w,m,n,p,q,cxv,alpha,beta,gamma,age]
                [UNIMPLEMENTED!]
                Represents a filament axisymmetric along longest position
                dimension, with isotropic velocity combined with extra
                component of bulk sheer orthogonal to longest position
                dimension.
                x,y,z,u,v,w : mean
                m : standard deviation in longest position dimension (X)
                n : standard deviation in the two (equal) smallest position
                    dimensions (Y,Z)
                p : standard deviation in the two (equal) smallest velocity
                    dimensions (U,W)
                q : standard deviation in the longest velocity dimension (V)
                cxv : correlation between X and V
                alpha,beta,gamma : the three Tait Bryan angles to perform
                    solid body rotation for arbitrary orientation
        internal : bool {False}
            Set to True if passing in parameters as emcee stores them. For
            example GroupFit fits to dx and dv in logspace.
        starcount : bool {False}
            [Maybe bad?]
            Set to True if final value in `pars` is the number of stars.
        """
        assert form in self.IMPLEMENTED_FORMS, 'provided form: {}'.format(form)
        self.form = form
        if pars is None:
            self.mean = mean
            self.covmatrix = covmatrix
            self.setParsFromMeanAndCov()

        else:
            logging.debug("Input: {}".format(pars))
            if internal:
                self.pars = self.externalisePars(pars, form=form)
            else:
                self.pars = pars
            self.form = form
            logging.debug("Interpreted: {}".format(self.pars))

            if self.form == 'sphere':
                self.mean = pars[:6]
                self.dx = self.pars[6]
                self.dv = self.pars[7]
                self.age = self.pars[8]

            elif self.form == 'elliptical':
                self.mean = pars[:6]
                self.dx, self.dy, self.dz = self.pars[6:9]
                self.dv = self.pars[9]
                self.cxy, self.cxz, self.cyz = self.pars[10:13]
                self.age = self.pars[13]

            # Construct cov matrix
            self.covmatrix = self.generateCovMatrix()

        # Set some general values based of CovMatrix
        self.sphere_dx = gmean(np.sqrt(
            np.linalg.eigvalsh(self.covmatrix[:3,:3]))
        )
        self.sphere_dv = gmean(np.sqrt(
            np.linalg.eigvalsh(self.covmatrix[3:,3:]))
        )


    def __eq__(self, other):
        """Predominantly implemented for testing reasons"""
        if isinstance(other, self.__class__):
            return np.allclose(self.pars, other.pars) and\
                   self.form == other.form
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def setParsFromMeanAndCov(self):
        """Using the mean and covariance matrix, establish most appropriatet
        pars, and update accordingly.
        Used primarily when a component is initialised by mean and covariance
        matrix.
        """
        if self.form == 'sphere':
            self.dx = self.sphere_dx = gmean(np.sqrt(
                np.linalg.eigvalsh(self.covmatrix[:3,:3]))
            )
            self.dv = self.sphere_dv = gmean(np.sqrt(
                    np.linalg.eigvalsh(self.covmatrix[3:,3:]))
            )
            self.age = self.DEFAULT_TINY_AGE

        elif self.form == 'elliptical':
            self.dv = self.sphere_dv = gmean(np.sqrt(
                    np.linalg.eigvalsh(self.covmatrix[3:,3:]))
            )
            xyz_covmatrix = self.covmatrix[:3,:3]
            stdevs = np.sqrt(np.diagonal(xyz_covmatrix))
            xyz_corrmatrix = xyz_covmatrix / stdevs / stdevs.reshape(1,3).T
            self.dx, self.dy, self.dz = stdevs
            self.cxy, self.cxz, self.cyz = xyz_corrmatrix[np.triu_indices(3,1)]
            self.age = 1e-5

        self.updatePars()

    def getInternalSphericalPars(self):
        """Build and return raw parametrisation of Component in
        internal, spherical form"""
        return np.hstack((self.mean, np.log(self.sphere_dx), np.log(self.dv),
                          self.age))

    def getSphericalPars(self):
        """Build and return raw parametrisation of Component in
        external, spherical form"""
        return np.hstack((self.mean, self.sphere_dx, self.dv, self.age))

    def getEllipticalPars(self):
        """Build and return raw parametrisation of Component in
        external, elliptical form"""
        if self.form == 'sphere':
            return np.hstack((self.mean, self.dx, self.dx, self.dx, self.dv,
                             0.0, 0.0, 0.0, self.age))
        elif self.form == 'elliptical':
            return np.copy(self.pars)

    def getPars(self, form=None):
        """Get pars in the form of the Component

        Parameters
        ----------
        form : str {None}
            Possible values: ['sphere'|'elliptical']
            If left as None, defaults to whatever form the Component
            was initialised as.

        Returns
        res : [n] float array_like
            Raw parameterisation of Component
        """
        if form is None:
            return np.copy(self.pars)
        elif form == 'sphere':
            return self.getSphericalPars()
        elif form == 'elliptical':
            return self.getEllipticalPars()
        else:
            raise ValueError


    def generateSphericalCovMatrix(self):
        """
        Build the initial covariance matrix based on spherical parameterisation.

        This covariance matrix is spherical in both position and velocity space,
        meaning there are no correlations between any axes, and the position
        standard deviations are all equal, and the velocity standard devaitions
        are all equal.

        Returns
        -------
        scmat : [6,6] float array_like
            The spherical covariance matrix of the Components origin
        """
        # Awkward checks to allow for two usages:
        # 1) initialise a spherical cov matrix from spherical pars input
        # 2) generate spherical equivalent of non-spherical covariance matrix
        if self.sphere_dx is None:
            dx = self.dx
        else:
            dx = self.sphere_dx
        if self.sphere_dv is None:
            dv = self.dv
        else:
            dv = self.sphere_dv
        scmat = np.array([
            [dx**2, 0., 0., 0., 0., 0.],
            [0., dx**2, 0., 0., 0., 0.],
            [0., 0., dx**2, 0., 0., 0.],
            [0., 0., 0., dv**2, 0., 0.],
            [0., 0., 0., 0., dv**2, 0.],
            [0., 0., 0., 0., 0., dv**2],
        ])
        return scmat

    def generateEllipticalCovMatrix(self):
        """
        Build the initial covariance matrix based on spherical parameterisation.

        This covariance matrix is spherical in both position and velocity space,
        meaning there are no correlations between any axes, and the position
        standard deviations are all equal, and the velocity standard devaitions
        are all equal.

        Returns
        -------
        scmat : [6,6] float array_like
            The spherical covariance matrix of the Components origin
        """
        # Handles scenario where component is not elliptical
        if self.form == 'sphere':
            return self.generateSphericalCovMatrix()

        dx, dy, dz = self.dx, self.dy, self.dz
        dv = self.dv
        cxy, cxz, cyz = self.cxy, self.cxz, self.cyz
        ecmat = np.array([
            [dx**2, cxy*dx*dy, cxz*dx*dz, 0., 0., 0.],
            [cxy*dx*dy, dy**2, cyz*dy*dz, 0., 0., 0.],
            [cxz*dx*dz, cyz*dy*dz, dz**2, 0., 0., 0.],
            [       0.,        0.,    0., dv**2, 0., 0.],
            [       0.,        0.,    0., 0., dv**2, 0.],
            [       0.,        0.,    0., 0., 0., dv**2],
        ])
        assert np.allclose(ecmat, ecmat.T)
        return ecmat

    def generateCovMatrix(self):
        """Builds and returns the initial covariance matrix based on
        parameterisation.

        Returns
        -------
        res : [6,6] float array_like
            The covariance matrix of a component's initial phase-space
            distribution.
        """
        if self.form == 'sphere':
            return self.generateSphericalCovMatrix()
        elif self.form == 'elliptical':
            return self.generateEllipticalCovMatrix()
        else:
            raise NotImplementedError

    def calcMeanNow(self):
        """
        Calculates the mean of the component when projected to the current-day
        """
        self.mean_now = traceorbit.traceOrbitXYZUVW(self.mean, times=self.age)

    def calcCovMatrixNow(self):
        """
        Calculates covariance matrix of current day distribution.

        Calculated as a first-order Taylor approximation of the coordinate
        transformation that takes the initial mean to the current day mean.
        """
        if self.mean_now is None:
            self.calcMeanNow()
        self.covmatrix_now = transform.transformCovMat(
            self.covmatrix, trans_func=traceorbit.traceOrbitXYZUVW,
            loc=self.mean, args=(self.age,),
        )

    def calcCurrentDayProjection(self):
        """
        Get the current day Gaussian distribution as a mean and cov matrix

        Returns
        -------
        mean_now : [6] float array
            The xyzuvw mean of the current day distribution
        covmatrix_now : [6,6] float array
            The covariance matrix of the current day distribution
        """
        if self.mean_now is None:
            self.calcMeanNow()
        if self.covmatrix_now is None:
            self.calcCovMatrixNow()


    def getCurrentDayProjection(self):
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
        self.calcCurrentDayProjection()
        return self.mean_now, self.covmatrix_now


    def updatePars(self):
        """
        Make sure `self.pars` field accurately maps to the various fields.

        This is useful to ensure consistency if fields are modified directly
        """
        if self.form == 'sphere':
            self.pars = np.hstack((self.mean, self.dx, self.dv, self.age))
        elif self.form == 'elliptical':
            self.pars = np.hstack((
                self.mean, self.dx, self.dy, self.dz, self.dv,
                self.cxy, self.cxz, self.cyz, self.age
            ))
        self.covmatrix = self.generateCovMatrix()

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
            covariance matrix but wiht an older age
        """
        if self.mean_now is None:
            self.calcMeanNow()

        comps = []
        for age in [lo_age, hi_age]:
            # base new component on current component, then modify fields
            new_comp = Component(self.pars, form=self.form)
            new_mean = traceorbit.traceOrbitXYZUVW(self.mean_now, times=-age)
            new_comp.mean = new_mean
            new_comp.age = age
            # Ensure new_comp.pars reflects the true parametrisation
            new_comp.updatePars()
            comps.append(new_comp)

        return comps

