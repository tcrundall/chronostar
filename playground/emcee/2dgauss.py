'''
The parameters for a model describing data has a
4D Gaussian probability surface.
'''
import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner

def calc_med_and_span(chain, perc=34):
    """
    Given a set of aligned samples, calculate the 50th, (50-perc)th and
     (50+perc)th percentiles.

    Parameters
    ----------
    chain : [nwalkers, nsteps, npars] array -or- [nwalkers*nsteps, npars] array
        The chain of samples (in internal encoding)
    perc: integer {34}
        The percentage from the midpoint you wish to set as the error.
        The default is to take the 16th and 84th percentile.

    Returns
    -------
    result : [npars,3] float array
        For each parameter, there is the 50th, (50+perc)th and (50-perc)th
        percentiles
    """
    npars = chain.shape[-1]  # will now also work on flatchain as input
    flat_chain = np.reshape(chain, (-1, npars))

    return np.array(list(map(lambda v: (v[1], v[2], v[0]),
                            zip(*np.percentile(flat_chain,
                                               [50-perc, 50, 50+perc],
                                               axis=0)))))

def approx_mean_and_std(chain):
    '''Approximate mean and standard deviation from chain'''
    med_and_span = calc_med_and_span(chain)
    means = med_and_span[:,0]
    stds = 0.5 * (med_and_span[:,1] - med_and_span[:,2])
    return np.vstack((means, stds)).T

def mvgauss(x, mean, cov, dim=4):
    '''Multivariate Gaussian'''
    det = np.linalg.det(cov)
    coeff = 1./np.sqrt((2*np.pi)**dim * det)
    offset = x - mean
    cov_inv = np.linalg.inv(cov)
    expon = -0.5 * np.dot(offset, np.dot(cov_inv, offset))
    return coeff * np.exp(expon)

def gauss(x, mu, sig, amp=1.):
    # print(sig)
    coeff = 1./np.sqrt(2*np.pi*sig**2)
    expon = -0.5 * (x-mu)**2 / sig**2
    return amp * coeff * np.exp(expon)

def lnprior(pars, a_mn, a_std, b_mn, b_std, amp=1.):
    # print(a_std, b_std)
    return (np.log(gauss(pars[0], a_mn, a_std, amp=amp)) +
                np.log(gauss(pars[1], b_mn, b_std, amp=amp)))

def lnlike(pars, surface_vals):
    surf_mn = surface_vals[:4]
    surf_std = surface_vals[4:8]
    corr_c_d = surface_vals[8]

    # Build up the covariance matrix
    surf_cov = np.eye(4)
    # Allowing for correlations between a or b and c or d breaks everything
    # surf_cov[0,3] = surf_cov[3,0] = 0.6
    # surf_cov[1,3] = surf_cov[3,1] = 0.6
    # surf_cov[1,2] = surf_cov[2,1] = 0.8
    surf_cov[2,3] = surf_cov[3,2] = corr_c_d
    surf_cov *= surf_std
    surf_cov *= surf_std[np.newaxis,:].T

    return np.log(mvgauss(pars, surf_mn, surf_cov))

def lnprob(pars, a_mn, a_std, b_mn, b_std, surface_vals, amp):
    return lnprior(pars, a_mn, a_std, b_mn, b_std, amp) +\
           lnlike(pars, surface_vals)

def test_mvgauss():
    import matplotlib.pyplot as plt
    npoints = 100
    lim = 8
    xs = np.linspace(-lim,lim,npoints)
    pars = np.zeros(4)
    scores = np.zeros(npoints)

    for par_ix in range(4):
        for ix, x in enumerate(xs):
            pars = np.zeros(4)
            pars[par_ix] = x
            scores[ix] = lnlike(pars)

        plt.clf()
        plt.plot(xs, scores)
        plt.savefig('{}_plot.png'.format(par_ix))

if __name__ == '__main__':
    # Setting the 4D mean and covariance for Gaussian surface
    #   A,    B,  C,  D
    surf_mns = np.array([
        20., 20., 0., 0.,
    ])
    #  dA, dB, dC, dD
    surf_stds = np.array([
        1000.,10., 3., 4.
    ])
    # maybe c and d have some interesting correlation
    corr_c_d = 0.2

    surface_vals = np.hstack((surf_mns, surf_stds, corr_c_d))

    # independently measured values of a and b
    a_mn, a_std = 0., 1.
    b_mn, b_std = 0., 1.

    # amplitude that amplifies the gaussian prior enforced by measured a and b
    amp = 1.e20

    burnin_steps = 100
    sampling_steps = 10000
    nwalkers = 10
    npars = 4

    init_centre = np.zeros(npars)
    init_std = np.ones(npars)
    init_pos = emcee.utils.sample_ball(init_centre, init_std, size=nwalkers)

    sampler = emcee.EnsembleSampler(
            nwalkers, npars, lnprob,
            args=[a_mn, a_std, b_mn, b_std, surface_vals, amp]
    )

    print('Amplifying measured Guass\'s by: {}'.format(amp))
    # burning
    p0, _, _ = sampler.run_mcmc(init_pos, burnin_steps)

    # reset and sample
    sampler.reset()
    sampler.run_mcmc(p0, sampling_steps)

    lnprob = sampler.lnprobability
    plt.clf()
    plt.plot(lnprob.T)
    plt.savefig('lnprobT.pdf')

    plt.clf()
    corner.corner(sampler.flatchain)
    plt.savefig('corner.pdf')

    np.save('chain.npy', sampler.flatchain)
    np.save('mean_and_errs.npy', approx_mean_and_std(sampler.flatchain))
    print(approx_mean_and_std(sampler.flatchain))

