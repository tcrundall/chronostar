"""
Issue is can't pickle a Class that has emcee sampler as an attribute?

To solve this, must simply have all functions emcee calls be external.
emcee functions can still utilise classes and objects (obvy)
"""


import numpy as np
import emcee


class Utils():
    def __init__(self):
        pass

    @staticmethod
    def class_return_val(val):
        print('hi')
        return val

    def return_val(self, val):
        print('hi 2')
        return val


def lnprior(theta):
    my_utils = Utils()
    theta = my_utils.return_val(theta)
    m, b = theta
    if -10.0 < m < 10.0 and 0.0 < b < 10.0:
        return 0.0
    return -np.inf


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    likelihood = lnlike(theta, x, y, yerr)
    return lp + likelihood


def lnlike(theta, x, y, yerr):
    m, b = theta
    model = m * x + b
    chi_squared = -0.5*np.sum((y-model)**2 / yerr)
    return chi_squared
    # return -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))


def run_mcmc(nstep, x, y, yerr):
    ndim, nwalkers = 2, 100
    # pos = [get_results(m, b, f, x, y, yerr) + 1e-4 * np.random.randn(ndim) for i in
    pos = [np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(x, y, yerr), threads=10)
    sampler.run_mcmc(pos, nstep)
    return sampler

class ModelFit():

    def __init__(self, x, y, yerr):
        pass
        self.x=x
        self.y=y
        self.yerr=yerr

    def perform_run(self, nstep):
        self.sampler = run_mcmc(nstep, self.x, self.y, self.yerr)

    @staticmethod
    def calc_median(sampler, cut_frac=0.5):
        """
        Have some useful, utility methods made static so samplers can be loaded
        from memory (for e.g.) and operated upon.
        """
        nwalkers, nsteps, dim = sampler.chain.shape
        cut_ix = int(cut_frac * nsteps)
        trimmed_flat_chain = sampler.chain[:, -cut_ix:].reshape(-1, dim)
        return np.median(trimmed_flat_chain, axis=0)

    def get_median(self, cut_frac=0.5):
        return self.calc_median(self.sampler, cut_frac=cut_frac)
