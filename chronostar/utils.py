"""
A selection of tools used by chronostar

TODO: Add unittest for internalise_pars
TODO: Add unittest for internalise_multi_pars
"""
from __future__ import division, print_function

import numpy as np
import pdb


def ix_fst(array, ix):
    if array is None:
        return None
    else:
        return array[ix]


def externalise_pars(pars_in, nstars=None):
    """Simple tool to convert pars in external form into internal form

    Parameters
    ----------
    pars_in : [X,Y,Z,U,V,W,1/dX,1/dY,1/dZ,1/dV,Cxy,Cxz,Cyz,age]

    Returns
    -------
    pars_ex : [X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars]
    """
    pars_ex = np.copy(pars_in)
    pars_ex[6:10] = 1/pars_ex[6:10]
    if nstars is not None:
        pars_ex = pars_ex.append(nstars)
    return pars_ex


def externalise_multi_pars(multi_pars_in, nstars=None):
    """Convert pars for multiple groups in external form into internal form

    Parameters
    ----------
    pars_in : [ngroups, 14] array
    nstars : [ngroups] array
        can optionally append nstars into the parameter list

    Returns
    -------
    pars_ex : [ngroups, 15 or 14] array
    """
    if nstars is not None:
        nstar_slot = 1
    else:
        nstar_slot = 0

    ngroups, npars = multi_pars_in.shape
    multi_pars_ex = np.zeros((ngroups, npars+nstar_slot))
    for i, pars_in in enumerate(multi_pars_in):
        multi_pars_ex[i] = externalise_pars(pars_in, ix_fst(nstars, i))
    return multi_pars_in


def internalise_pars(pars_ex):
    """Simple tool to convert pars in external form into internal form

    Parameters
    ----------
    pars_ex : [X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars]

    Returns
    -------
    pars_in : [X,Y,Z,U,V,W,1/dX,1/dY,1/dZ,1/dV,Cxy,Cxz,Cyz,age]
    """
    pars_in = np.copy(pars_ex[:-1])
    pars_in[6:10] = 1/pars_in[6:10]
    return pars_in


def internalise_multi_pars(multi_pars_ex):
    """Convert pars for multiple groups in external form into internal form

    Parameters
    ----------
    pars_ex : [ngroups, 15] array

    Returns
    -------
    pars_in : [ngroups, 14] array
    """
    ngroups, npars = multi_pars_ex.shape
    multi_pars_in = np.zeros((ngroups, npars-1))
    for i, pars_ex in enumerate(multi_pars_ex):
        multi_pars_in[i] = internalise_pars(pars_ex)
    return multi_pars_in


def approx_spread(cov_matrix):
    """Approximate the spread of a cov_matrix in position space
    """
    pos_cov = cov_matrix[:3, :3]
    eig_vals = np.sqrt(np.linalg.eigvalsh(pos_cov))
    return (np.prod(eig_vals)) ** (1.0 / 3.0)


def approx_spread_from_sample(sample):
    """Approximate the spread of a sample in position space

    Parameters
    ----------
    sample : [14] array
        emcee parameters in interal encoding (1/dX, no nstars, etc)
    """
    cov = generate_cov(sample)
    return approx_spread(cov)


def approx_spread_from_chain(chain):
    """Approximate the spread of emcee samples in XYZ space"""
    if len(chain.shape) > 2:
        chain = chain.reshape((-1, chain.shape[-1]))
    nsamples = chain.shape[0]

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
    dX, dY, dZ, dV = 1.0 / np.array(pars[6:10])
    Cxy, Cxz, Cyz = pars[10:13]
    cov = np.array([
        [dX ** 2, Cxy * dX * dY, Cxz * dX * dZ, 0.0, 0.0, 0.0],
        [Cxy * dX * dY, dY ** 2, Cyz * dY * dZ, 0.0, 0.0, 0.0],
        [Cxz * dX * dZ, Cyz * dY * dZ, dZ ** 2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, dV ** 2, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, dV ** 2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, dV ** 2],
    ])
    return cov
