
import numpy as np
import matplotlib.pyplot as plt
import pdb


def x_from_pol(r, theta):
    return r*np.cos(theta)


def y_from_pol(r, theta):
    return r*np.sin(theta)


def transform_ptcs(rs, thetas):
    xs = x_from_pol(rs, thetas)
    ys = y_from_pol(rs, thetas)
    return np.array([xs, ys]).T


def transform_ptc(loc):
    x = x_from_pol(loc[0], loc[1])
    y = y_from_pol(loc[0], loc[1])
    return np.array([x,y])


r_mean = np.sqrt(10**2+ 10**2)
theta_mean = 14*np.pi / 6.

r_std = 1.
theta_std = np.pi / 24.

C_rt = 0.0

pol_mean = np.array([r_mean, theta_mean])
pol_cov = np.array([
    [r_std**2, C_rt * r_std * theta_std],
    [C_rt * r_std * theta_std, theta_std**2],
])

nsamples = 100000
pol_samples = np.random.multivariate_normal(pol_mean, pol_cov, nsamples)

res = transform_ptcs(pol_samples[:, 0], pol_samples[:, 1])

cart_mean = transform_ptcs(*pol_mean)

def get_jac_col(trans_func, col_number, loc, dim=2, h=1e-3):
    """
    Calculate a column of the Jacobian.

    A whole column can be done in one hit because we are incrementing
    the same parameter of the initial coordinate system.

    :param trans_func:
        Transformation function taking us from the initial coordinate frame
        to the final coordinate frame
    :param col_number:
        The index in question (which parameter of the intial frame we are
        incrementing
    :param loc:
        The position (in the initial coordinte frame) around which we are
        calculting the jacobian
    :param dim: [2]
        The dimensionality of the coordinate frames.
    :param h: [1e-3]
        The size of the increment
    :return: The column

    """
    offset = np.zeros(dim)
    offset[col_number] = h
    loc_pl = loc + offset
    loc_mi = loc - offset

    return (trans_func(loc_pl) - trans_func(loc_mi)) / (2*h)

def get_jac(trans_func, loc, dim=2, h=1e-3):
    """

    :param trans_func:
        Transformation function taking us from the initial coordinate frame
        to the final coordinate frame
    :param loc:
        The position (in the initial coordinte frame) around which we are
        calculting the jacobian
    :param dim:
        The dimensionality of the coordinate frames
    :param h:
        The size of the increment
    :return: A jacobian
    """
    jac = np.zeros((dim, dim))
    for i in range(dim):
        jac[:,i] = get_jac_col(trans_func, i, loc, dim, h)

    return jac

def transform_cov(cov, trans_func, loc, dim=2):
    """
    Transforming a covariance matrix from one coordinate frame to another

    :param cov:
        Covariance matrix in the initial frame
    :param trans_func:
        Transformation function taking us from the initial coordinate frame
        to the final coordinate frame
    :param loc:
        The position (in the initial coordinte frame) around which we are
        calculting the jacobian
    :param dim:
        The dimensionality of the coordinate frames
    :return:
    """
    jac = get_jac(trans_func, loc, dim=2)
    return np.dot(jac, np.dot(cov, jac.T))

jac_from_func = get_jac(transform_ptc, pol_mean)

h = 1e-3
# loc 0,0
mean_plus = pol_mean + np.array([h, 0.])
mean_minus = pol_mean - np.array([h, 0.])
first_jac_col =\
    (transform_ptcs(mean_plus[0], mean_plus[1])
     -transform_ptcs(mean_minus[0], mean_minus[1])) / (2*h)

mean_plus = pol_mean + np.array([0., h])
mean_minus = pol_mean - np.array([0., h])
second_jac_col = \
    (transform_ptcs(mean_plus[0], mean_plus[1])
     -transform_ptcs(mean_minus[0], mean_minus[1])) / (2*h)

jac = np.zeros((2,2))
jac[:,0] = first_jac_col
jac[:,1] = second_jac_col

cart_mean = transform_ptc(pol_mean)
#jac = get_jac(transform_ptc, pol_mean)
cart_cov  = transform_cov(pol_cov, transform_ptc, pol_mean, dim=2)

cart_samples = np.random.multivariate_normal(cart_mean, cart_cov, nsamples)

estimated_mean = np.mean(res, axis=0)
estimated_cov = np.cov(res.T)


if True:
    nbins = 100
    plt.clf()
    plt.hist2d(res[:,0],res[:,1],bins=nbins,range=((-20,20),(-20,20)))
    plt.savefig("temp_jac_hist.png")

    plt.clf()
    plt.plot(res[:,0], res[:,1], '.')
    plt.ylim(-20,20)
    plt.xlim(-20,20)
    plt.savefig("temp_jac.png")

    plt.clf()
    plt.plot(cart_samples[:,0], cart_samples[:,1], '.')
    plt.ylim(-20,20)
    plt.xlim(-20,20)
    plt.savefig("temp_jac_cart.png")

    plt.clf()
    plt.hist2d(cart_samples[:,0],cart_samples[:,1],bins=nbins,range=((-20,20),(-20,20)))
    plt.savefig("temp_jac_cart_hist.png")

# tolerance is so high because dealing with non-linearity and
assert np.allclose(estimated_mean, cart_mean, rtol=1e-1)
assert np.allclose(estimated_cov, cart_cov, rtol=1e-1)
