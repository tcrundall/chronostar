import logging
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '..')

#from chronostar.retired.tracingback import trace_forward
#from chronostar.retired import utils
import chronostar.traceorbit as torb
from chronostar.synthesiser import Group
import chronostar.errorellipse as ee
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


def polar_demo():
    """
    Demonstrating the use of the transform module in covariance transformations
    between coordinate systems
    """
    logging.info("Beginning polar demo")
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
    cart_cov  = tf.transformCovMat(pol_cov, transform_ptc, pol_mean, dim=2)

    cart_samples = np.random.multivariate_normal(cart_mean, cart_cov, nsamples)

    estimated_mean = np.mean(res, axis=0)
    estimated_cov = np.cov(res.T)

    # plotting
    if plotit:
        nbins = 100
        plt.clf()
        plt.hist2d(res[:,0],res[:,1],bins=nbins,range=((-20,20),(-20,20)))
        plt.savefig("temp_plots/jac_hist.png")

        plt.clf()
        plt.plot(res[:,0], res[:,1], '.')
        plt.ylim(-20,20)
        plt.xlim(-20,20)
        plt.savefig("temp_plots/jac.png")

        plt.clf()
        plt.plot(cart_samples[:,0], cart_samples[:,1], '.')
        plt.ylim(-20,20)
        plt.xlim(-20,20)
        plt.savefig("temp_plots/jac_cart.png")

        plt.clf()
        plt.hist2d(cart_samples[:,0],cart_samples[:,1],bins=nbins,range=((-20,20),(-20,20)))
        plt.savefig("temp_plots/jac_cart_hist.png")

    # tolerance is so high because dealing with non-linearity and
    assert np.allclose(estimated_mean, cart_mean, rtol=1e-1)
    assert np.allclose(estimated_cov, cart_cov, rtol=1e-1)
    logging.info("Ending polar demo")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    polar_demo()
    plotit = True
    #              X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars

    nstars = 100
    age = 20.

    dummy_groups = [
        #X,Y,Z,U,V,W,dX,dY,dZ, dV,Cxy,Cxz,Cyz,age,
        [0,0,0,0,0,0,10,10,10,  2, 0.5, 0.2, 0.,age, nstars], # isotropic expansion
        [0,0,0,0,0,0,10, 1, 1, .1, 0., 0., 0.,2*age, nstars], # should rotate anticlock
        [-20,-20,300,0,0,0,10,10,10,  2, 0., 0., 0.,age, nstars], # isotropic expansion
        [0,0,0,0,10,0,10, 1, 1, .1, 0., 0., 0., 1000, nstars],
    ]

    my_group = Group(dummy_groups[0], sphere=False)

    #for cnt, dummy_group_pars_ex in enumerate(dummy_groups):
    mean = my_group.mean
    cov  = my_group.generateEllipticalCovMatrix()

    stars = np.random.multivariate_normal(mean, cov, nstars)
    if plotit:
        plt.clf()
        plt.plot(stars[:,0], stars[:,1], 'b.')
        #plt.hist2d(stars[:,0], stars[:,1], bins=20)
        ee.plotCovEllipse(cov[:2, :2], mean, color='b', alpha=0.3)

    new_stars = np.zeros(stars.shape)
    new_stars = torb.traceManyOrbitXYZUVW(stars, np.array(0, age))

def stop():
    # calculate the new mean and cov
    new_mean = trace_forward(mean, age)
    new_cov = tf.transformCovMat(cov, trace_forward, mean, dim=6, args=(age,))
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
        ee.plotCovEllipse(
            new_cov[:2,:2], new_mean, color='r', alpha=0.1
        )
        plt.savefig("temp_plots/trace_forward{}.png".format(cnt))
        plt.show()

    assert np.allclose(new_eigvals, estimated_eigvals, rtol=.5)

