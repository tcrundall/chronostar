import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
sys.path.insert(0, '..')

from chronostar.traceback import trace_forward
import chronostar.traceback as tb
from chronostar import utils
import chronostar.error_ellipse as ee
import chronostar.transform as tf


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

#
#
#def get_jac_col(trans_func, col_number, loc, dim=2, h=1e-3, args=None):
#    """
#    Calculate a column of the Jacobian.
#
#    A whole column can be done in one hit because we are incrementing
#    the same parameter of the initial coordinate system.
#
#    :param trans_func:
#        Transformation function taking us from the initial coordinate frame
#        to the final coordinate frame
#    :param col_number:
#        The index in question (which parameter of the intial frame we are
#        incrementing
#    :param loc:
#        The position (in the initial coordinte frame) around which we are
#        calculting the jacobian
#    :param dim: [2]
#        The dimensionality of the coordinate frames.
#    :param h: [1e-3]
#        The size of the increment
#    :return: The column
#
#    """
#    offset = np.zeros(dim)
#    offset[col_number] = h
#    loc_pl = loc + offset
#    loc_mi = loc - offset
#    if args is None:
#        return (trans_func(loc_pl) - trans_func(loc_mi)) / (2*h)
#    else:
#        return (trans_func(loc_pl, *args) - trans_func(loc_mi, *args)) / (2*h)
#
#def get_jac(trans_func, loc, dim=2, h=1e-3, args=None):
#    """
#
#    :param trans_func:
#        Transformation function taking us from the initial coordinate frame
#        to the final coordinate frame
#    :param loc:
#        The position (in the initial coordinte frame) around which we are
#        calculting the jacobian
#    :param dim:
#        The dimensionality of the coordinate frames
#    :param h:
#        The size of the increment
#    :return: A jacobian
#    """
#    jac = np.zeros((dim, dim))
#    for i in range(dim):
#        jac[:,i] = get_jac_col(trans_func, i, loc, dim, h, args)
#
#    return jac
#
#def transform_cov(cov, trans_func, loc, dim=2, args=None):
#    """
#    Transforming a covariance matrix from one coordinate frame to another
#
#    :param cov:
#        Covariance matrix in the initial frame
#    :param trans_func:
#        Transformation function taking us from the initial coordinate frame
#        to the final coordinate frame
#    :param loc:
#        The position (in the initial coordinte frame) around which we are
#        calculting the jacobian
#    :param dim:
#        The dimensionality of the coordinate frames
#    :return:
#    """
#    jac = get_jac(trans_func, loc, dim=dim, args=args)
#    return np.dot(jac, np.dot(cov, jac.T))

def polar_demo():
    plotit = True
    # setting up polar points
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

    # converting to cartesian manually for comparison
    res = transform_ptcs(pol_samples[:, 0], pol_samples[:, 1])

    cart_mean = transform_ptc(pol_mean)
    cart_cov  = tf.transform_cov(pol_cov, transform_ptc, pol_mean, dim=2)

    cart_samples = np.random.multivariate_normal(cart_mean, cart_cov, nsamples)

    estimated_mean = np.mean(res, axis=0)
    estimated_cov = np.cov(res.T)

    # plotting
    if plotit:
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


if __name__ == '__main__':
    polar_demo()
    plotit = True
    func = trace_forward

    #              X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars

    nstars = 100
    age = 20.

    dummy_groups = [
        #X,Y,Z,U,V,W,dX,dY,dZ, dV,Cxy,Cxz,Cyz,age,
        [0,0,0,0,0,0,10,10,10,  2, 0., 0., 0.,age], # isotropic expansion
        [0,0,0,0,0,0,10, 1, 1, .1, 0., 0., 0.,2*age], # should rotate anticlock
        [-20,-20,300,0,0,0,10,10,10,  2, 0., 0., 0.,age], # isotropic expansion
    ]

    for cnt, dummy_group_pars_ex in enumerate(dummy_groups):
        mean = dummy_group_pars_ex[0:6]
        cov  = utils.generate_cov(
                    utils.internalise_pars(dummy_group_pars_ex)
                )
        stars = np.random.multivariate_normal(mean, cov, nstars)
        if plotit:
            plt.clf()
            plt.plot(stars[:,0], stars[:,1], 'b.')
            #plt.hist2d(stars[:,0], stars[:,1], bins=20)
            ee.plot_cov_ellipse(cov[:2,:2], mean)
            #plt.show()

        new_stars = np.zeros(stars.shape)
        for i, star in enumerate(stars):
            new_stars[i] = trace_forward(star, age)

        # calculate the new mean and cov
        new_mean = trace_forward(mean, age)
        new_cov = tf.transform_cov(cov, trace_forward, mean, dim=6, args=(age,))
        import pdb; pdb.set_trace()
        new_eigvals = np.linalg.eigvalsh(new_cov)

        estimated_cov = np.cov(new_stars.T)
        estimated_eigvals = np.linalg.eigvalsh(estimated_cov)


        if plotit:
            #plt.clf()
            plt.plot(new_stars[:,0], new_stars[:,1], 'r.')
            #plt.hist2d(stars[:,0], stars[:,1], bins=20)
            ymin, ymax = plt.ylim()
            xmin, xmax = plt.xlim()

            upper = max(xmax, ymax)
            lower = min(xmin, ymin)

            plt.xlim(upper, lower)
            plt.ylim(lower, upper)
            ee.plot_cov_ellipse(
                new_cov[:2,:2], new_mean, color='r', alpha=0.1
            )
            plt.savefig("temp_trace_forward{}.png".format(cnt))
            plt.show()

        assert np.allclose(new_eigvals, estimated_eigvals, rtol=.5)
