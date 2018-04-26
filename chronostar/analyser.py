
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
