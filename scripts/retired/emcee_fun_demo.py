from __future__ import division, print_function

"""
Just a bit of emcee fun
"""

import emcee
import corner
import numpy as np
import logging
import matplotlib.pyplot as plt
import sys

ASA_SET = np.array([
    # A
    [2,5],
    [2,6],
    [2,7],
    [2,8],
    [2,9],
    [3,6],
    [3,9],
    [4,5],
    [4,6],
    [4,7],
    [4,8],
    [4,9],

    # S
    [6,5],
    [6,7],
    [6,8],
    [6,9],
    [7,5],
    [7,7],
    [7,9],
    [8,5],
    [8,6],
    [8,7],
    [8,9],

    # A
    [10,5],
    [10,6],
    [10,7],
    [10,8],
    [10,9],
    [11,6],
    [11,9],
    [12,5],
    [12,6],
    [12,7],
    [12,8],
    [12,9],

    # 2
    [1, -2],
    [1, -1],
    [1, 0],
    [1, 2],
    [2, -2],
    [2, 0],
    [2, 2],
    [3, -2],
    [3, 0],
    [3, 1],
    [3, 2],

    # 0
    [5, -2],
    [5, -1],
    [5, 0],
    [5, 1],
    [5, 2],
    [6, -2],
    [6, 2],
    [7, -2],
    [7, -1],
    [7, 0],
    [7, 1],
    [7, 2],

    # 1
    [9, -2],
    [9, -1],
    [9, 0],
    [9, 1],
    [9, 2],

    # 8
    [11, -2],
    [11, -1],
    [11, 0],
    [11, 1],
    [11, 2],
    [12, -2],
    [12, 0],
    [12, 2],
    [13, -2],
    [13, -1],
    [13, 0],
    [13, 1],
    [13, 2],
])

def checkPointInSet(test_point, set=ASA_SET, atol=0.5):
    for point in set:
        if np.allclose(test_point, point, atol=atol):
            return True
    return False


def burninConvergence(lnprob, tol=0.25, slice_size=100, cutoff=0):
    """Checks early lnprob vals with final lnprob vals for convergence

    Parameters
    ----------
    lnprob : [nwalkers, nsteps] array

    tol : float
        The number of standard deviations the final mean lnprob should be
        within of the initial mean lnprob

    slice_size : int
        Number of steps at each end to use for mean lnprob calcultions

    cuttoff : int
        Step number at which to start the analysis. i.e., a value of 0 would
        mean to include the whole chain, whereas a value of 500 would be to
        only confirm no deviation in the steps [500:END]
    """
    # take a chunk the smaller of 100 or half the chain
    if lnprob.shape[1] < slice_size:
        slice_size = int(round(0.5*lnprob.shape[1]))

    start_lnprob_mn = np.mean(lnprob[:,:slice_size])
    #start_lnprob_std = np.std(lnprob[:,:slice_size])

    end_lnprob_mn = np.mean(lnprob[:, -slice_size:])
    end_lnprob_std = np.std(lnprob[:, -slice_size:])

    return np.isclose(start_lnprob_mn, end_lnprob_mn,
                      atol=tol*end_lnprob_std)


def asaLnprobFunc(pars):
    x,y = pars
    if (x < -2 or x > 15 or y < -5 or y > 12):
        return -np.inf
    elif (checkPointInSet(pars)):
        return 2
    else:
        return -1

def faceLnprobFunc(pars,):
    """Computes the log-probability for a fit to a group.

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
        0,1,2,3,4,5,   6,   7,  8
        X,Y,Z,U,V,W,lndX,lndV,age
    star_pars
        traceback data being fitted to
    (retired) z
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    Returns
    -------
    logprob
        the logarithm of the posterior probability of the fit
    """

    x, y = pars

    face_r = 5.
    eye_r = 0.7
    eye_y = 1.5
    eye_x = 2.

    # set circular boundary for face
    if (x**2 + y**2 >= face_r**2):
        return -np.inf

    # inserting eyes
    if ((x-eye_x)**2 + (y-eye_y)**2 <= eye_r**2 or
            (x+eye_x)**2 + (y-eye_y)**2 <= eye_r**2):
        return 2.

    # inserting mouth
    if ( (x > -3 and x < 3 and y > -2.5 and y < -1.85) ):
        return 2.

    # return -np.inf
    return -1.


def acquirePDF(plot_dir='temp_plots/', save_dir='temp_data/',
             burnin_steps=300, sampling_steps=300, plot_it=True):
    """Fits a single gaussian to a weighted set of traceback orbits.

    Parameters
    ----------
    plot_dir : str {''}
        The directory in which to save plots

    save_dir : str {''}
        The directory in which to store results and/or byproducts of fit

    init_pars : [npars] array
        the position in parameter space about which walkers should be
        initialised. The standard deviation about each parameter is
        hardcoded as INIT_SDEV

    sampling_steps : int {None}
        If this is set, after convergence, a sampling stage will be
        entered. Only do this if a very fine map of the parameter
        distributions is required, since the burnin stage already
        characterises a converged solution for "burnin_steps".

    Returns
    -------
    best_sample
        The parameters of the group model which yielded the highest
        posterior probability

    chain
        [nwalkers, nsteps, npars] array of all samples

    probability
        [nwalkers, nsteps] array of probabilities for each sample
    """
    init_pars = [0.,0.]
    INIT_SDEV = [2.,2.]
    # init_pars = [6.,6.]
    # INIT_SDEV = [2.,2.]

    NPAR = len(init_pars)
    NWALKERS = 2 * 2 * NPAR


    # Whole emcee shebang
    sampler = emcee.EnsembleSampler(NWALKERS, NPAR, faceLnprobFunc)

    # initialise walkers, note that INIT_SDEV is carefully chosen such thata
    # all generated positions are permitted by lnprior
    logging.info("Initialising walkers around the parameters:\n{}".\
                 format(init_pars))
    logging.info("With standard deviation:\n{}".\
                 format(INIT_SDEV) )
    pos = np.array([
        init_pars + (np.random.random(size=len(INIT_SDEV)) - 0.5)\
        * INIT_SDEV
        for _ in range(NWALKERS)
    ])

    # Perform burnin
    logging.info("Beginning burnin for {} steps".format(burnin_steps))
    sampler.reset()
    pos, lnprob, state = sampler.run_mcmc(pos, burnin_steps)

    # Help out the struggling walkers
    best_ix = np.argmax(lnprob)
    poor_ixs = np.where(lnprob < np.percentile(lnprob, 33))
    for ix in poor_ixs:
        pos[ix] = pos[best_ix]

    if plot_it:
        plt.clf()
        plt.plot(sampler.lnprobability.T)
        plt.savefig(plot_dir+"face_burnin_lnprobT.png")

    logging.info("Entering sampling stage for {} steps".format(
        sampling_steps
    ))
    sampler.reset()
    pos, lnprob, state = sampler.run_mcmc(pos, sampling_steps, state)
    logging.info("Sampling done")

    # save the chain for later inspection
    np.save(save_dir+"face_final_chain.npy", sampler.chain)
    np.save(save_dir+"face_final_lnprob.npy", sampler.lnprobability)

#    print("Sampled")
    if plot_it:
        logging.info("Plotting final lnprob")
        plt.clf()
        plt.plot(sampler.lnprobability.T)
        plt.savefig(plot_dir+"face_lnprobT.png")
#        logging.info("Plotting corner")
#        plt.clf()
#        corner.corner(sampler.flatchain)
#        plt.savefig(plot_dir+"corner.pdf")
        logging.info("Plotting done")

    # sampler.lnprobability has shape (NWALKERS, SAMPLE_STEPS)
    # yet np.argmax takes index of flattened array
    final_best_ix = np.argmax(sampler.lnprobability)
    best_sample = sampler.flatchain[final_best_ix]

    plt.clf()
    corner.corner(sampler.flatchain, bins=32)
    plt.savefig(plot_dir + "face_corner.pdf")

    return best_sample, sampler.chain, sampler.lnprobability

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    acquirePDF(burnin_steps=2000, sampling_steps=1000000, plot_dir='temp_plots/')

