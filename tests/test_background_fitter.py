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

