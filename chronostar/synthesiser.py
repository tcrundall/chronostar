"""
synthesiser

Used to generate realistic data for one (or many) synthetic association(s)
with multiple starburst events along with a background as desired.

From a parametrised gaussian distribution, generate the starting
XYZUVW values for a given number of stars

TODO: accommodate multiple groups
"""

from __future__ import print_function, division

import logging
import numpy as np

class Group:
    def __init__(self, pars, sphere=True, internal=False, starcount=True):
        # If sphere flag is set, interpret pars one way
        # If not set, interpret pars another way
        # Simply supposed to be a neat way of packaging up a group's initial
        # conditions
        logging.debug("Input: {}".format(pars))
        self.pars = np.copy(pars)

        if sphere:
            nstdevs = 2
        else:
            nstdevs = 4
        if internal:
            # convert logarithms into linear vals
            self.pars[6:6+nstdevs] = np.exp(self.pars[6:6+nstdevs])

        logging.debug("Interpreted: {}".format(self.pars))
        self.is_sphere = sphere
        if sphere:
            self.mean = pars[:6]
            self.dx = self.sphere_dx = self.pars[6]
            self.dv = self.pars[7]
            self.age = self.pars[8]
        else:
            self.mean = pars[:6]
            self.dx, self.dy, self.dz = self.pars[6:9]
            self.dv = self.pars[9]
            self.cxy, self.cxz, self.cyz = self.pars[10:13]
            self.age = self.pars[13]

            self.sphere_dx = (self.dx * self.dy * self.dz)**(1./3.)
        if starcount:
            self.nstars = int(self.pars[-1])
        else:
            self.nstars = None
            logging.debug("No star count provided")

    def __eq__(self, other):
        """Predominantly implemented for testing reasons"""
        if isinstance(other, self.__class__):
            return np.allclose(self.pars, other.pars) and\
                   self.is_sphere == other.is_sphere
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def getInternalSphericalPars(self):
        return np.hstack((self.mean, np.log(self.sphere_dx), np.log(self.dv),
                          self.age))

    def getSphericalPars(self):
        return np.hstack((self.mean, self.sphere_dx, self.dv, self.age))


    def getFreePars(self):
        if self.is_sphere:
            return np.hstack((self.mean, self.dx, self.dx, self.dx, self.dv,
                             0.0, 0.0, 0.0, self.age))
        else:
            return np.hstack((self.mean, self.dx, self.dy, self.dz, self.dv,
                             self.cxy, self.cxz, self.cyz, self.age))

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
        if self.is_sphere:
            return self.generateSphericalCovMatrix()
        else:
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
        if self.is_sphere:
            return self.generateSphericalCovMatrix()
        else:
            return self.generateEllipticalCovMatrix()

def synthesiseXYZUVW(pars, sphere=True, return_group=False,
                     xyzuvw_savefile='', group_savefile='', internal=False):
    """
    Generate a bunch of stars in situ based off a Guassian parametrisation

    Parameters
    ----------
    pars : [10] or [15] float array
        10 parameters : [X,Y,Z,U,V,W,dX,dV,age,nstars]
            Covariance matrix describes a spherical distribution in pos
            and vel space
        15 parameters : [X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars]
            Covariance matrix descirbes a spherical distribution in velocity
            space and freely orientable, triaxial ellipsoid in position space
    sphere : boolean {True}
        Set flag True if providing pars in 9 parameter form,
        Set flag False if providing pars in 14 parameter form,
    return_group : boolean {False}
        Set flag if want to return the group object (for tracking input
        parameters)
    xyzuvw_savefile : String {''}
        Provide a string to numpy.save the init_xyzuvw array
    group_savefile : Stirng {''}
        Provide a string to numpy.save the group object; note you need to
        np.load(group_savefile).item() in order to retrieve it.
    internal : Boolean {False}
        Set if parameters are provided in emcee internalised form

    Returns
    -------
    xyzuvw_init : [nstars,6] float array
        Initial distribution of stars in XYZUVW coordinates in corotating, RH
        (X,U positive towards galactic anti-centre) cartesian coordinates
        centred on local standard fo rest.

    (if flag return_group is set)
    group : SynthGroup object
        An object that wraps initialisation parameters
    """
    logging.debug("Internal?: {}".format(internal))
    group = Group(pars, sphere=sphere, internal=internal)
    logging.debug("Mean {}".format(group.mean))
    logging.debug("Cov\n{}".format(group.generateCovMatrix()))
    logging.debug("Number of stars {}".format(group.nstars))
    init_xyzuvw = np.random.multivariate_normal(
        mean=group.mean, cov=group.generateCovMatrix(),
        size=group.nstars,
    )
    if xyzuvw_savefile:
        np.save(xyzuvw_savefile, init_xyzuvw)
    if group_savefile:
        np.save(group_savefile, group)
    if return_group:
        return init_xyzuvw, group
    else:
        return init_xyzuvw

def synthesiseManyXYZUVW(many_pars, sphere=True, return_groups=False,
                         xyzuvw_savefile='', groups_savefile='',
                         internal=False):
    """
    Generate a bunch of stars in situ from many Gaussian parametrisations

    Note: there is no orbital projection, only the initial positions
    of the stars are returned
    As a consequence, this function is stupid and useless....

    Parameters
    ----------
    many_pars : [ngroups, 10] or [ngroups, 15] float array
        10 parameters : [X,Y,Z,U,V,W,dX,dV,age,nstars]
            Covariance matrix describes a spherical distribution in pos
            and vel space
        15 parameters : [X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars]
            Covariance matrix descirbes a spherical distribution in velocity
            space and freely orientable, triaxial ellipsoid in position space
    sphere : boolean {True}
        Set flag True if providing pars in 9 parameter form,
        Set flag False if providing pars in 14 parameter form,
    return_groups : boolean {False}
        Set flag if want to return the group object (for tracking input
        parameters)
    xyzuvw_savefile : String {''}
        Provide a string to numpy.save the init_xyzuvw array
    groups_savefile : Stirng {''}
        Provide a string to numpy.save the group object; note you need to
        np.load(group_savefile).item() in order to retrieve it.
    internal : Boolean {False}
        Set if parameters are provided in emcee internalised form

    Returns
    -------
    xyzuvw_init : [Nstars,6] float array
        Initial distribution of stars in XYZUVW coordinates in corotating, RH
        (X,U positive towards galactic anti-centre) cartesian coordinates
        centred on local standard fo rest.
        Nstars is the sum of all group pars' nstars

    (if flag return_group is set)
    groups : [ngroups] Group object
        Objects that wrap initialisation parameters
    """
    many_pars_cp = np.copy(many_pars)

    groups = []
    all_init_xyzuvw = np.zeros((0,6))

    for pars in many_pars_cp:
        init_xyzuvw, group = synthesiseXYZUVW(pars, sphere=sphere,
                                              return_group=True,
                                              internal=internal)
        groups.append(group)
        all_init_xyzuvw = np.vstack((all_init_xyzuvw, init_xyzuvw))
    if xyzuvw_savefile:
        np.save(xyzuvw_savefile, all_init_xyzuvw)
    if groups_savefile:
        np.save(groups_savefile, groups)
    if return_groups:
        return all_init_xyzuvw, groups
    else:
        return all_init_xyzuvw


