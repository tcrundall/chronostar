"""
A module which aids in the transformation of a covariance matrix
between two coordinate frames. This is used explicitly for the
trace_forward_group code in the traceback module.
There is scope to incorporate this into the traceback code too, for
clarity and ease of reading
"""

import numpy as np

def get_jac_col(trans_func, col_number, loc, dim=2, h=1e-3, args=None):
    """
    Calculate a column of the Jacobian.

    A whole column can be done in one hit because we are incrementing
    the same parameter of the initial coordinate system. This used to be
    accurate to the first order by taking the difference of a forward increment
    with a backward increment, but trans_func is often expensive. So now we
    just take the difference with the forward increment with the current
    position.

    Parameters
    ----------
    trans_func: (function)
        Transformation function taking us from the initial coordinate frame
        to the final coordinate frame
    col_number: (integer)
        The index in question (which parameter of the intial frame we are
        incrementing
    loc: ([dim] float array)
        The position (in the initial coordinte frame) around which we are
        calculting the jacobian
    dim: (integer {2})
        The dimensionality of the coordinate frames.
    h: (float {1e-3})
        The size of the increment
    :return: The column

    """
    offset = np.zeros(dim)
    offset[col_number] = h
    loc_pl = loc + offset
    loc_mi = loc - offset
    if args is None:
        return (trans_func(loc_pl) - trans_func(loc_mi)) / (2*h)
    else:
        return (trans_func(loc_pl, *args) - trans_func(loc_mi, *args)) / (2*h)

def get_jac(trans_func, loc, dim=2, h=1e-3, args=None):
    """

    trans_func:
        Transformation function taking us from the initial coordinate frame
        to the final coordinate frame
    loc:
        The position (in the initial coordinte frame) around which we are
        calculting the jacobian
    dim:
        The dimensionality of the coordinate frames
    h:
        The size of the increment

    Returns
    -------
    jac: ([dim,dim] float array)
        A jacobian matrix
    """
    jac = np.zeros((dim, dim))
    for i in range(dim):
        jac[:,i] = get_jac_col(trans_func, i, loc, dim, h, args)

    return jac

def transform_cov(cov, trans_func, loc, dim=6, args=None):
    """
    Transforming a covariance matrix from one coordinate frame to another

    Paramters
    ---------
    cov: ([dim,dim] float array)
        Covariance matrix in the initial frame
    trans_func: (function)
        Transformation function taking us from the initial
        coordinate frame to the final coordinate frame. Output must be
        mutable, i.e. single value, or an array
    loc: ([dim] float array)
        The position (in the initial coordinate frame)
        around which we are calculting the jacobian
    dim: (integer {6})
        The dimensionality of the coordinate frames

    Returns
    -------
    conv_cov: ([dim,dim] float array)
        The transformed covariance matrix
    """
    jac = get_jac(trans_func, loc, dim=dim, args=args)
    return np.dot(jac, np.dot(cov, jac.T))
