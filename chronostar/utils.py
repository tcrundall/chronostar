"""
A selection of tools used by chronostar
"""
from __future__ import division, print_function

import numpy as np
import pdb

def approx_spread(cov_matrix):
    """Approximate the spread of a cov_matrix in position space
    """
    pos_cov = cov_matrix[:3,:3]
    eig_vals = np.sqrt(np.linalg.eigvalsh(pos_cov))
    return (np.prod(eig_vals))**(1.0/3.0)

def approx_spread_from_chain(chain):
    if len(chain.shape) > 2:
        chain = chain.reshape((-1,chain.shape[-1]))
    nsamples = chain.shape[0]

    cov_matrices = np.zeros((nsamples, 6, 6))
    cov_spreads = np.zeros((nsamples))
    for i in range(nsamples):
        cov_matrix = generate_cov(chain[i])
        cov_spreads[i] = approx_spread(cov_matrix)
    return np.mean(cov_spreads)

def generate_cov(pars):
    """Generate covariance matrix from standard devs and correlations

    Parameters
    ----------
    pars
        [14] array with the following values:
        pars[6:10] : [1/dX, 1/dY, 1/dZ, 1/dV] :
            standard deviations in position and velocity
            for group model or stellar PDFs
        pars[10:13] : [CorrXY, CorrXZ, CorrYZ]
            correlations between position

    Returns
    -------
    cov
        [6, 6] array : covariance matrix for group model or stellar pdf
    """
    dX, dY, dZ, dV = 1.0/np.array(pars[6:10])
    Cxy, Cxz, Cyz  = pars[10:13]
    cov = np.array([
            [dX**2,     Cxy*dX*dY, Cxz*dX*dZ, 0.0,   0.0,   0.0],
            [Cxy*dX*dY, dY**2,     Cyz*dY*dZ, 0.0,   0.0,   0.0],
            [Cxz*dX*dZ, Cyz*dY*dZ, dZ**2,     0.0,   0.0,   0.0],
            [0.0,       0.0,       0.0,       dV**2, 0.0,   0.0],
            [0.0,       0.0,       0.0,       0.0,   dV**2, 0.0],
            [0.0,       0.0,       0.0,       0.0,   0.0,   dV**2],
        ])
    return cov


