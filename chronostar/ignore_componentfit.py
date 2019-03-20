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
import emcee

from component import Component
from . import tabletool
from likelihood import lnprob_func

class ComponentFit():
    """
    Captures the fit of a single component fit.

    Relies heavily on the Component class. If you desire a different
    parameterisation of the initial conditions we suggest you extend the
    component class with a new "form" input. So long as there is a means to
    develop a multivariate Gaussian with an initial mean and covariance matrix
    then the model can be incorporated.
    """

    def __init__(self, data, membership_probs=None, maximiser='emcee',
                 sampler='emcee', model_form='sphere'):
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
        self.maximiser = maximiser
        self.sampler = sampler
        self.model_form = model_form
        self.ismaximised = False


    def checkIfMaximised(self, **kwargs):
        if self.maximiser == 'emcee':
            if 'slice_size' in kwargs.keys():
                slice_size = kwargs['slice_size']
            else:
                slice_size = 100
            lnprob = kwargs['lnprob']
            mean_at_start = np.mean(lnprob[:, :slice_size])

            mean_at_end = np.mean(lnprob[:, -slice_size:])
            std_at_end = np.std(lnprob[:, -slice_size:])
            self.ismaximised = (np.isclose(mean_at_start, mean_at_end,
                                           atol=self.tol*std_at_end))


    def maximise(self, **kwargs):
        """
        Find (hopefully) the global maximum of the likelihood function
        """
        if self.maximiser == 'emcee':
            kwargs = {
                'steps':200,
                'state':None,
            }
            while not self.ismaximised:
                print("maximising...")
                kwargs = self.runEmcee(**kwargs)
        else:
            raise UserWarning('emcee is sole implementation of maximise')

    def sample(self, best_sample=None, **kwargs):
        """
        Given the best_sample (or prev best pos from `maximise` run) sample
        the region to establish uncertainty of best parameters
        """
        if self.maximiser == 'emcee':
            self.runEmcee(**kwargs)
        else:
            raise UserWarning('emcee is sole implementation of sample')


    def approxCurrentDayDistribution(self):
        means = tabletool.build_data_dict_from_table(self.data, cartesian=True,
                                                     only_means=True)
        mean_of_means = np.average(means, axis=0, weights=self.membership_probs)
        cov_of_means = np.cov(means.T, ddof=0., aweights=self.membership_probs)
        return mean_of_means, cov_of_means


    def getInitEmceePos(self, nwalkers=None, init_pars=None):
        if init_pars is None:
            rough_mean_now, rough_cov_now = \
                self.approxCurrentDayDistribution()
            # Exploit the component logic to generate closest set of pars
            dummy_comp = Component(mean=rough_mean_now,
                                   covmatrix=rough_cov_now,
                                   form=self.model_form)
            init_pars = Component.internalisePars(dummy_comp.getPars())

        init_std = Component.getSensibleInitSpread(form='sphere')

        # Generate initial positions of all walkers by adding some random
        # offset to `init_pars`
        if nwalkers is None:
            npars = len(init_pars)
            nwalkers = 2 * npars
        init_pos = emcee.utils.sample_ball(init_pars, init_std,
                                           size=nwalkers)
        # force ages to be positive
        init_pos[:, -1] = abs(init_pos[:, -1])
        return init_pos


    def runEmcee(self, nwalkers=None, init_pos=None,
                 init_pars=None, **kwargs):
        """Run an emcee fit given the data and model"""
        # If not provided, establish the initial position of the walkers
        if init_pos is None:
            # If not provided, centre swarm of walkers on data mean
            init_pos = self.getInitEmceePos(nwalkers=nwalkers,
                                            init_pars=init_pars)
        nwalkers, npars = init_pos.shape
        sampler = emcee.EnsembleSampler(nwalkers, npars, lnprob_func,
                                        args=[self.data, self.membership_probs])

        pos, lnprob, state = sampler.run_mcmc(pos=init_pos,
                                              steps=kwargs['steps'],
                                              state=kwargs['state'])
        res = {
            'init_pos':pos,
            'lnprob':lnprob,
            'state':state,
        }
        return res










