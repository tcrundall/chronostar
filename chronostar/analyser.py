
import logging
import numpy as np

import synthesiser as syn


def getBestSample(chain, lnprob):
    """
    Finds the sample with the highest lnprob

    Parameters
    ----------
    chain : [nsteps, nwalkers, npars] float array
    lnprob : [nsteps, nwalkers] float array
    """
    npars = chain.shape[-1]
    best_ix = np.argmax(lnprob)
    flat_chain = chain.reshape(-1, npars)

    best_sample = flat_chain[best_ix]

    if npars == 9:
        sphere = True
    elif npars == 15:
        sphere = False
    else:
        raise ValueError

    best_group = syn.Group(best_sample, sphere=sphere, internal=True)
    return best_group

def calcMedianWithErrs(flat_samples):
    """
    Given a set of aligned (converted?) samples, calculate the median and
    errors of each parameter

    Parameters
    ----------
    flat_samples : [nwalkers*nsteps, npars] float array
        the flat chain (sampler.chain.reshape(-1,npars) from the sampling stage

    Returns
    -------
    [npars, 3] float array
        For each parameter, returns the median, +error, -error
        where the errors represent the 84th and 16th percentile
    """
    return np.array( map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                     zip(*np.percentile(flat_samples, [16,50,84], axis=0))))


