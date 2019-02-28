"""
Class object that encapsulates a component, the phase-space model
of an unbound set of stars formed from the same starburst/filament.
"""

from __future__ import print_function, division, unicode_literals

import logging
import numpy as np

from . import transform
from . import traceorbit

class Component:
    IMPLEMENTED_FORMS = ('sphere', 'elliptical')

    mean_now = None
    covmatrix_now = None

    @staticmethod
    def loadComponents(filename):
        res = np.load(filename)
        if len(res) == 0:
            return res.item()
        else:
            return res

    @staticmethod
    def externalisePars(pars, form='sphere'):
        """
        Convert parameters from internal form to external form.
        """
        pars = np.copy(pars)
        if form == 'sphere':
            pars[6:8] = np.exp(pars[6:8])
        elif form == 'elliptical':
            pars[6:10] = np.exp(pars[6:10])
        return pars


    def __init__(self, pars, form='sphere', internal=False):
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
        logging.debug("Input: {}".format(pars))
        if internal:
            self.pars = self.externalisePars(pars, form=form)
        else:
            self.pars = pars
        self.form = form
        logging.debug("Interpreted: {}".format(self.pars))

        if self.form == 'sphere':
            self.mean = pars[:6]
            self.dx = self.sphere_dx = self.pars[6]
            self.dv = self.pars[7]
            self.age = self.pars[8]

        elif self.form == 'elliptical':
            self.mean = pars[:6]
            self.dx, self.dy, self.dz = self.pars[6:9]
            self.dv = self.pars[9]
            self.cxy, self.cxz, self.cyz = self.pars[10:13]
            self.age = self.pars[13]

            self.sphere_dx = (self.dx * self.dy * self.dz)**(1./3.)

        self.covmatrix = self.generateCovMatrix()

    def __eq__(self, other):
        """Predominantly implemented for testing reasons"""
        if isinstance(other, self.__class__):
            return np.allclose(self.pars, other.pars) and\
                   self.form == other.form
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def getInternalSphericalPars(self):
        return np.hstack((self.mean, np.log(self.sphere_dx), np.log(self.dv),
                          self.age))

    def getSphericalPars(self):
        return np.hstack((self.mean, self.sphere_dx, self.dv, self.age))

    def getEllipticalPars(self):
        if self.form == 'sphere':
            return np.hstack((self.mean, self.dx, self.dx, self.dx, self.dv,
                             0.0, 0.0, 0.0, self.age))
        elif self.form == 'elliptical':
            return self.pars
            # return np.hstack((self.mean, self.dx, self.dy, self.dz, self.dv,
            #                  self.cxy, self.cxz, self.cyz, self.age))

    def generateSphericalCovMatrix(self):
        dx = self.sphere_dx
        dv = self.dv
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
        # if self.is_sphere:
        #     return self.generateSphericalCovMatrix()
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
        if self.form == 'sphere':
            return self.generateSphericalCovMatrix()
        elif self.form == 'elliptical':
            return self.generateEllipticalCovMatrix()
        else:
            raise NotImplementedError

    def calcMeanNow(self):
        self.mean_now = traceorbit.traceOrbitXYZUVW(self.mean, times=self.age)

    def calcCovMatrixNow(self):
        """
        Calculate covariance matrix of current day distribution.

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
        self.calcCurrentDayProjection()
        return self.mean_now, self.covmatrix_now


    def updatePars(self):
        """
        Make sure `self.pars` field accurately maps to the various fields.

        This is useful to ensure consistency if fields are modfied direclty
        """
        if self.form == 'sphere':
            self.pars = np.hstack((self.mean, self.dx, self.dv, self.age))
        elif self.form == 'eliptical':
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

