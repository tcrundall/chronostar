import logging
import numpy as np
import sys

sys.path.insert(0, '..')

import chronostar.synthesiser as syn
import scripts.background_fitter as bf

def test_calc_MLE_mean():
    npoints = 100000
    true_mean = np.array([10.,10.,5.])
    true_cov_mat = np.array([
        [100., 25., -5],
        [25., 40., 0.],
        [-5., 0.,  25.]
    ])

    rand_data = np.random.multivariate_normal(true_mean, true_cov_mat, npoints)
    np_mean = rand_data.mean(axis=0)
    np_cov = np.cov(rand_data.T)

    assert np.allclose(true_mean, np_mean, rtol=0.1)
    assert np.allclose(true_cov_mat, np_cov, rtol=0.1, atol=0.2)

    mle_mean = bf.calc_MLE_mean(rand_data, np.ones((npoints,1)))
    assert np.allclose(true_mean, mle_mean, rtol=0.1)
    mle_cov = bf.calc_MLE_cov(rand_data, np.ones((npoints,1)), mle_mean)
    print(mle_cov)
    assert np.allclose(true_cov_mat, mle_cov, rtol=0.1, atol=0.2)


def test_einsum_ln_eval():
    nstars = 3
    gaia_data = np.load("../data/gaia_dr2_mean_xyzuvw.npy")[:nstars]
    ncomps = 1
    means = np.random.rand(ncomps,6) * 100
    covs = ncomps * [np.diag((100,100,100,20,20,20))]
    z = np.random.rand(nstars, ncomps)

    ln_evals_new = bf.evalLnMvgaussEinsum(gaia_data, means, covs, z)

    ln_evals_orig = np.zeros((gaia_data.shape[0], ncomps))
    for i, (mu, sigma) in enumerate(zip(means, covs)):
        sigma_inv = np.linalg.inv(sigma)
        sigma_det = np.linalg.det(sigma)
        weight = z[:,i].sum()
        for j in range(nstars):
            ln_evals_orig[j,i] = np.log(weight) +\
                bf.eval_ln_mvgauss(gaia_data[j], mu, sigma,
                                   sigma_inv, sigma_det)

    ln_evals_orig_orig = bf.calc_ln_eval(gaia_data, means, covs, z)

    # import pdb; pdb.set_trace()
    # call with -s flag to show print statements
    print(ln_evals_orig_orig)
    print(ln_evals_orig)
    print(ln_evals_new)
    print(ln_evals_new - ln_evals_orig)
    assert np.allclose(ln_evals_new, ln_evals_orig_orig, atol=0.1)

