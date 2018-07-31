"""
Fits many 6D Gaussians to Gaia background catalogue using
 a simple Expectation Maximisation algorithm
"""
from __future__ import print_function, division
import logging
import matplotlib.pyplot as plt
import numpy as np
import pdb
import time

import sys
sys.path.insert(0, '..')
import chronostar.expectmax as em
import chronostar.synthesiser as syn


#def plot_cov_ellipse(cov, pos, volume=.5, ax=None, fc='none', ec=[0, 0, 0], a=1,
#                     lw=2):
#    """
#    Plots an ellipse enclosing *volume* based on the specified covariance
#    matrix (*cov*) and location (*pos*). Additional keyword arguments are
#    passed on to the
#    ellipse patch artist.
#
#    Parameters
#    ----------
#        cov : The 2x2 covariance matrix to base the ellipse on
#        pos : The location of the center of the ellipse. Expects a 2-element
#            sequence of [x0, y0].
#        volume : The volume inside the ellipse; defaults to 0.5
#        ax : The axis that the ellipse will be plotted on. Defaults to the
#            current axis.
#    """
#    import numpy as np
#    from scipy.stats import chi2
#    import matplotlib.pyplot as plt
#    from matplotlib.patches import Ellipse
#
#    def eigsorted(cov):
#        vals, vecs = np.linalg.eigh(cov)
#        order = vals.argsort()[::-1]
#        return vals[order], vecs[:, order]
#
#    if ax is None:
#        ax = plt.gca()
#
#    vals, vecs = eigsorted(cov)
#    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
#
#    kwrg = {'facecolor': fc, 'edgecolor': ec, 'alpha': a, 'linewidth': lw}
#
#    # Width and height are "full" widths, not radius
#    width, height = 2 * np.sqrt(chi2.ppf(volume, 2)) * np.sqrt(vals)
#    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)
#
#    ax.add_artist(ellip)
#
#
#def plot_components(mu, Sigma, colours, *args, **kwargs):
#    '''
#    Plot ellipses for the bivariate normals with mean mu[:,i] and covariance
#    Sigma[:,:,i]
#    '''
#    assert mu.shape[0] == Sigma.shape[0]
#    assert mu.shape[1] == 2
#    assert Sigma.shape[1] == 2
#    assert Sigma.shape[2] == 2
#    for i in range(mu.shape[1]):
#        kwargs['ec'] = colours[i]
#        plot_cov_ellipse(Sigma[i, :, :], mu[i, :], *args, **kwargs)
#
#
#import matplotlib.colors as mcol
#
#br_cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["b", "r"])
#
#
#def plot_data(redness=None):
#    if redness is not None:
#        assert len(redness) == data.shape[1]
#        assert all(_ >= 0 and _ <= 1 for _ in redness)
#        c = redness
#    else:
#        c = 'g'
#    plt.figure(figsize=(8, 8))
#    plt.scatter(data[0, :], data[1, :], marker='.', s=8, linewidths=2, c=c,
#                cmap=br_cmap)
#    plt.xlabel(data_labels[0])
#    plt.ylabel(data_labels[1])
#    #plt.axis([-2, 2, -2, 2], 'equal')

def calc_MLE_mean(data, z, comp_ix=0):
    """
    Finds mean of Gaussian fit with membership weightings

    Paramters
    ---------
    data : [nstars,6] array
        the central estimates of stars
    z : [nstars, ncomp] array
        membership array

    Returns
    -------
    mean : [6] array
    """
    ept_count = z[:,comp_ix].sum()
    mean = np.einsum('i,ij->j',z[:,comp_ix],data) / ept_count
    return mean

def calc_MLE_cov(data, z, mean, comp_ix=0):
    """
    Calculate the membership weighted covariance matrix of fit

    Paramters
    ---------
    data : [nstars,6] array
        the central estimates of stars
    z : [nstars, ncomp] array
        membership array
    comp_ix : int
        the index of the component currently being inspected

    Return
    ------
    cov_mat : [6,6] array
    """
    ept_count = np.sum(z[:,comp_ix])
    diff = data - mean
    cov_mat = np.einsum('i,ij,ik->jk',z[:,comp_ix],diff,diff) / ept_count
    return cov_mat

def eval_ln_mvgauss(d, mu, sigma, sigma_inv, s_det):
#    d = np.array(d).reshape(-1,1)
#    mu = np.array(mu).reshape(-1,1)
    k = len(d)
    #sigma = np.array(sigma).reshape(k,k)
    # pdb.set_trace()
    coeff = 1./np.sqrt(np.linalg.det(2*np.pi*sigma))
    expon = -0.5*np.dot(d - mu, np.dot(sigma_inv, d - mu))
    return np.log(coeff) + expon


def evalLnMvgaussEinsum(data, mus, sigmas, z):
    weights = np.sum(z, axis=0)
    nstars = z.shape[0]
    ncomps = z.shape[1]
    ln_evals = np.zeros((nstars, ncomps))
    for j in range(ncomps):
        sigma_inv = np.linalg.inv(sigmas[j])
        diff = data - mus[j]
        ln_evals[:,j] = np.einsum('im,mn,in->i',diff,sigma_inv,diff)
        coeff = np.log(np.linalg.det(2 * np.pi * sigmas[j]))
        ln_evals[:,j] += coeff
    ln_evals *= -0.5
    for j in range(ncomps):
        ln_evals[:,j] += np.log(weights[j])
    return ln_evals


def calcMembershipProbs(star_ln_evals):
    """Calculate probabilities of membership for a single star from overlaps

    Parameters
    ----------
    star_ln_evals : [ngroups] array
        The log of the evaluation of a star with each group

    Returns
    -------
    star_memb_probs : [ngroups] array
        The probability of membership to each group, normalised to sum to 1
    """
    #logging.info("Calculating membership probs")
    ngroups = star_ln_evals.shape[0]
    star_memb_probs = np.zeros(ngroups)

    for i in range(ngroups):
        star_memb_probs[i] = 1. / np.sum(np.exp(star_ln_evals - star_ln_evals[i]))

    #logging.info("done")
    return star_memb_probs


def calc_ln_eval(data, mus, sigmas, old_z):
    ncomps = old_z.shape[1]
    nstars = old_z.shape[0]
    ln_evals = np.zeros((data.shape[0], ncomps))

    if old_z is None:
        old_z = np.ones((nstars, ncomps))/ncomps

    for i, (mu, sigma) in enumerate(zip(mus, sigmas)):
        sigma_inv = np.linalg.inv(sigma)
        sigma_det = np.linalg.det(sigma)
        weight = old_z[:,i].sum()
        for j in range(nstars):
#            if (j%100000) == 0:
#               print("Evaluated {} of {}".format(j, nstars))
            ln_evals[j,i] = np.log(weight) + eval_ln_mvgauss(data[j], mu,
                                                             sigma, sigma_inv,
                                                             sigma_det)
    return ln_evals

def e_step(data, mus, sigmas, old_z=None):
    e_start = time.time()
    nstars = len(data)
    ncomps = len(mus)

    if old_z is None:
        old_z = np.ones((nstars, ncomps))/ncomps

    #ln_evals = calc_ln_eval(data, mus, sigmas, old_z)
    ln_evals = evalLnMvgaussEinsum(data, mus, sigmas, old_z)

    #pdb.set_trace()
    z = np.zeros((nstars, ncomps))
    for i in range(nstars):
        z[i] = calcMembershipProbs(ln_evals[i])
    if np.isnan(z).any():
        logging.info("!!!!!! AT LEAST ONE MEMBERSHIP IS 'NAN' !!!!!!")
        #import pdb; pdb.set_trace()
    e_end = time.time()
    logging.info("E Time taken: {:02.0f}:{:05.2f}".\
                 format(int((e_end - e_start) // 60), (e_end - e_start)%60))
    return z, ln_evals

def m_step(data, z):
    m_start = time.time()
    means = []
    cov_mats = []
    ncomps = z.shape[1]

    for i in range(ncomps):
        mean = calc_MLE_mean(data, z, i)
        cov_mat = calc_MLE_cov(data, z, mean, i)
        means.append(mean)
        cov_mats.append(cov_mat)
        if np.isnan(mean).any():
            print("!!!!!! AT LEAST ONE MEAN IS 'NAN' !!!!!!")
            #import pdb; pdb.set_trace()

    m_end = time.time()
    logging.info("M Time taken: {:02.0f}:{:05.2f}". \
                 format(int((m_end - m_start) // 60), (m_end - m_start)%60))
    return means, cov_mats


def calc_lnlike(z, ln_evals):
    # weights = z.sum(axis=0)

    # nstars = z.shape[0]
    # ncomps = z.shape[1]

    # cov_mat_dets = []
    # cov_mat_invs = []
    # for cov_mat in cov_mats:
    #     cov_mat_dets.append(np.linalg.det(cov_mat))
    #     cov_mat_invs.append(np.linalg.inv(cov_mat))

    # ln_evals = evalLnMvgaussEinsum(data, means, cov_mats, z)
    lnlike = np.einsum('ij,ij->',ln_evals,z)

    # lnlike = 0
    # for i in range(nstars):
    #     lnlike_contrib = 0
    #     for j in range(ncomps):
    #         # pdb.set_trace()
    #         lnlike_contrib += z[i,j] *\
    #                           eval_ln_mvgauss(
    #                               data[i], means[j], cov_mats[j],
    #                               cov_mat_invs[j], cov_mat_dets[j]
    #                           )
    #     lnlike += lnlike_contrib
    return lnlike

def initialise_means(data, ncomps):
    groups = em.getInitialGroups(ncomps, data, offset=False)

    means = []
    cov_mats = []
    for group in groups:
        means.append(group.mean)
        cov_mats.append(group.generateCovMatrix())
    return means, cov_mats

if __name__ == '__main__':
    ITERATIONS = 100
    NSTARS = 100000
    ncomps = 2
    start = time.time()
    logging.basicConfig(level=logging.DEBUG,
                        filename="bgfitter_{}_{}.log".format(ncomps,
                                                             ITERATIONS),
                        filemode='w')
    logging.info("Initialising run")
    logging.info("Iterations: {}".format(ITERATIONS))
    logging.info("N components: {}".format(ncomps))
    logging.info("Up to star index: {}".format(NSTARS))

    rdir = "../results/background_fits/"
    mean_savefile = rdir + "bg_means_{}.npy".format(ncomps)
    cov_savefile = rdir + "bg_covs_{}.npy".format(ncomps)
    z_savefile = rdir + "z_{}.npy".format(ncomps)
    logging.info("Storing results in {}".format(rdir))

    lnlikes = []

    gaia_file = "../data/gaia_dr2_mean_xyzuvw.npy"
    try:
        gaia_xyzuvw = np.load(gaia_file)[:NSTARS]
        logging.info("Nstars are: {}".format(gaia_xyzuvw.shape[0]))
    except IOError:
        gaia_file = "/data/mash/tcrun/gaia_dr2_mean_xyzuvw.npy"
        gaia_xyzuvw = np.load(gaia_file)[:NSTARS]
    # gaia_xyzuvw = np.loadtxt('../data/faithful.csv', delimiter=',')
    # gaia_xyzuvw -= gaia_xyzuvw.mean(axis=0)
    # gaia_xyzuvw /= gaia_xyzuvw.std(axis=0)
    data_labels = ('Eruption length','Eruption wait')
    data = gaia_xyzuvw.T

    nstars = gaia_xyzuvw.shape[0]
    #z = np.ones((nstars, ncomps))
    #    z = np.random.rand(nstars, ncomps)
    #    print("z shape: {}".format(z.shape))
    #    inv_sum = np.sum(z, axis=1)**-1
    #    z = np.einsum('i,ij->ij',inv_sum, z)
    #    print(np.sum(z, axis=0))
    #    print("z shape: {}".format(z.shape))
    #means, cov_mats = m_step(gaia_xyzuvw, z)

    try:
        means = np.load(mean_savefile)
        cov_mats = np.load(cov_savefile)
        logging.info("Loaded fit from previous run")
    except IOError:
        logging.info("Initialsing from scratch")
        means, cov_mats = initialise_means(gaia_xyzuvw, ncomps)
    # means = [np.array([1,-1]), np.array([-1,1])]
    # cov_mats = [np.diag((2,1)), np.diag((1,2))]
    #pdb.set_trace()
    for i in range(ITERATIONS):
        iter_start = time.time()
        logging.info("---- Iteration {} of {} -----".format(i, ITERATIONS))
        old_means = np.copy(means)
        new_z, ln_evals = e_step(gaia_xyzuvw, means, cov_mats)
        logging.info("Current weight breakdown:\n{}"\
                     .format(np.sum(new_z, axis=0)))
        means, cov_mats = m_step(gaia_xyzuvw, new_z)
        means_diff = 100*np.mean(abs(
            (np.array(means)-np.array(old_means))/np.array(old_means)
        ))
        np.save(mean_savefile, np.array(means))
        np.save(cov_savefile, np.array(cov_mats))
        np.save(z_savefile, np.array(new_z))


        logging.info("{:.4f}% difference in means".format(means_diff))

        lnlike_start = time.time()
        lnlikes.append(calc_lnlike(new_z, ln_evals))

        lnlike_end = time.time()
        logging.info("lnlike time taken: {:02.0f}:{:05.2f}" \
                     .format(int((lnlike_end - lnlike_start) // 60),
                             (lnlike_end - lnlike_start)%60))

        logging.info("Lnlike: {}".format(lnlikes[-1]))

        iter_end = time.time()
        logging.info("Iter time taken: {:02.0f}:{:05.2f}"\
                     .format(int((iter_end - iter_start) // 60),
                             (iter_end - iter_start)%60))

#        if not i % 10:
#            pdb.set_trace()
#            plot_data(redness=new_z[:,1])
#            plot_components(np.array(means)[:,:2], np.array(cov_mats)[:,:2,:2],
#                            ['b', 'r'], 0.2)
#            plt.show()
    plt.clf()
    plt.plot(lnlikes)
    plt.savefig("temp_plots/bgfitting_lnlike_{}_{}.pdf".\
                format(ncomps, ITERATIONS))
    np.save(rdir+"bg_lnlike_{}.npy".format(ncomps), np.array(lnlikes))

    end = time.time()
    logging.info("_________ COMPLETE __________")
    logging.info("Time taken: {:02.0f}:{:05.2f}".format(int((end - start) // 60),
                                         (end - start)%60))
