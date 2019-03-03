"""
Encapsulates fitting a single component to a set of cartesian kinematic data
(XYZUVW) paired with membership probabilities.

Notes
-----
Set things up such that data, model, maximiser and sampler can be plugged in

Data: table/dictionary:
    'X', 'Y', 'Z', 'U', 'V', 'W',
    'dX', 'dY', ...
    'c_XY', 'c_XZ', ...

Model: Gaussian form
    Some parametrisation of the initial distribution of kinematic data. Needs
    two parts: (1) raw parametrisation convention; (2) some machinery
    to convert from raw pars to Gaussian input, and; (3) means to get current
    day distribution

Posterior function:
    Means to generate some score on the "goodness of fit" from the data, model
    and a sample set of parameters.

Maximiser:
    Algorithm to find the best fitting parameters

Sampler:
    Algorithm to explore the PDF of parameters around the global maximum
"""
from astropy.table import Table
import numpy as np

from . import component

class ComponentFit():
    """
    Captures the fit of a single component fit.

    Relies heavily on the Component class. If you desire a different
    parameterisation of the initial conditions we suggest you extend the
    component class with a new "form" input. So long as there is a means to
    develop a multivariate Gaussian with an initial mean and covariance matrix
    then the model can be incorporated.


    """

    def __init__(self, data, membership_probs=None):
        """
        Parameters
        ----------
        data : astropy Table -or- string
            A table (or path to table) with kinematic data of stars with
            nstars rows
        membership_probs : [nstars] float array_like {None}
            An array of floats in the range 0.0 to 1.0 representing probability
            of provided stars being members of the association.
        """

        # Handle inputs
        if isinstance(data, str):
            self.data = Table.read(data)
        else:
            self.data = data
        self.nstars = len(data)
        if membership_probs is None:
            self.membership_probs = np.ones(len(self.nstars))
        else:
            self.membership_probs = membership_probs


    def run(self):
        """Run an emcee fit given the data and model"""