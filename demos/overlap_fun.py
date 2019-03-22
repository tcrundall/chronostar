import pdb
import logging
import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.compfitter as gf
from chronostar.likelihood import slow_get_lnoverlaps as sclno
import chronostar._overlap as ol

def clno(A, a, B, b):
    """
    Equivalent to co2, but with logarithms
    """
    logging.debug("---------------------------------------------------")
    logging.debug("clno(): In python implmentation of Tim's derivation")
    logging.debug("---------------------------------------------------")
    #logging.debug("Inputs are:")
    #logging.debug("  A:\n{}".format(A))
    #logging.debug("  a:\n{}".format(a))
    #logging.debug("  B:\n{}".format(B))
    #logging.debug("  b:\n{}".format(b))
    ApB = (A + B)
    logging.debug("ApB: \n{}".format(ApB))
    ApB_det = np.linalg.det(ApB)

    ApB_i = np.linalg.inv(ApB)
    amb = a - b

    result = 6 * np.log(2*np.pi)
    logging.debug("Added 6log(2pi): {}".format(result))
    lndet = np.log(ApB_det)
    logging.debug("Log of det(ApB): {}".format(lndet))
    result += np.log(ApB_det)
    logging.debug("result so far: {}".format(result))
    exponent = (np.dot(amb, np.dot(ApB_i, amb)))
    logging.debug("matrix exponent is: {}".format(exponent))
    result += exponent
    logging.debug("result so far: {}".format(result))
    result *= -0.5

#    lnoverlap = -0.5 * (np.dot(amb, np.dot(ApB_i, amb)))
#    lnoverlap -= 3 * np.log(2 * np.pi)
#    lnoverlap -= 0.5 * np.log(ApB_det)

    #logging.debug("Final result is: {}".format(lnoverlap))

    logging.debug("---------------------------------------------------")
    logging.debug("Final result is: {}".format(result))
    logging.debug("---------------------------------------------------")
    return result


def co2(A, a, B, b):
    """
    This is an alternative derivation of the overlap integral between
    two multivariate gaussians. This is the version implemented
    in the swigged C module.
    """

    ApB = (A + B)
    ApB_det = np.linalg.det(ApB)

    ApB_i = np.linalg.inv(ApB)

    # amn_m_bmn = a_mn - b_mn

    overlap = np.exp(-0.5 * (np.dot(a - b, np.dot(ApB_i, a - b))))
    overlap *= 1.0 / ((2 * np.pi) ** 3.0 * np.sqrt(ApB_det) )

    return overlap

def co1(A_cov, a, B_cov, b):
    """
    The original python function written by Mike yeaaaaarrss ago
    """
    A = np.linalg.inv(A_cov)
    B = np.linalg.inv(B_cov)
    A_det = np.linalg.det(A)
    B_det = np.linalg.det(B)

    # Preliminaries - add matrices together. This might make code more
    # readable?
    # Or might not.
    ApB = A + B
    AapBb = np.dot(A, a) + np.dot(B, b)

    # Compute determinants.
    ApB_det = np.linalg.det(ApB)

    # Error checking (not needed in C once shown to work?) This shouldn't
    # ever happen, as
    # the determinants of the sum of positive definite matrices is
    # greater than the sum of their determinants
    if (ApB_det < 0) | (B_det < 0):
        pdb.set_trace()
        return -np.inf

    # Solve for c
    c = np.linalg.solve(ApB, AapBb)

    # Compute the overlap formula.
    overlap = np.exp(-0.5 * (np.dot(b - c, np.dot(B, b - c)) + \
                             np.dot(a - c, np.dot(A, a - c))))
    overlap *= np.sqrt(B_det * A_det / ApB_det) / (2 * np.pi) ** 3.0
    return overlap



def test_pythonFuncs():
    xyzuvw_file = "../data/fed_stars_20_xyzuvw.fits"
    xyzuvw_dict = gf.loadXYZUVW(xyzuvw_file)

    star_means = xyzuvw_dict['xyzuvw']
    star_covs = xyzuvw_dict['xyzuvw_cov']
    nstars = star_means.shape[0]

    group_mean = np.mean(star_means, axis=0)
    group_cov = np.cov(star_means.T)

    co1s = []
    co2s = []
    for i, (scov, smn) in enumerate(zip(star_covs, star_means)):
        print(i)
        co1s.append(co1(group_cov, group_mean, scov, smn))
        co2s.append(co2(group_cov, group_mean, scov, smn))
        ol.get_lnoverlaps(group_cov, group_mean,
                          np.array([scov]),
                          np.array([smn]), 1)
    co1s = np.array(co1s)
    co2s = np.array(co2s)
    co3s = np.exp(sclno(group_cov, group_mean, star_covs, star_means, nstars))
    assert np.allclose(co1s, co2s)
    assert np.allclose(co2s, co3s)
    assert np.allclose(co1s, co3s)

    # note that most overlaps go to 0, but the log overlaps retains the
    # information
    co1s = []
    co2s = []
    for i, (scov, smn) in enumerate(zip(star_covs, star_means)):
        co1s.append(co1(star_covs[15], star_means[15], scov, smn))
        co2s.append(co2(star_covs[15], star_means[15], scov, smn))
    co1s = np.array(co1s)
    co2s = np.array(co2s)
    lnos = sclno(star_covs[15], star_means[15], star_covs, star_means, 1)
    co3s = np.exp(lnos)
    assert np.allclose(co1s, co2s)
    assert np.allclose(co2s, co3s)
    assert np.allclose(co1s, co3s)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    DEBUG_SINGULAR = False
    DEBUG_UNDERFLOW = True
    co1s = []
    co2s = []
    xyzuvw_file = "../data/fed_stars_20_xyzuvw.fits"
    xyzuvw_dict = gf.loadXYZUVW(xyzuvw_file)

    star_means = xyzuvw_dict['xyzuvw']
    star_covs = xyzuvw_dict['xyzuvw_cov']
    nstars = star_means.shape[0]

    gmn = np.mean(star_means, axis=0)
    gcov = np.cov(star_means.T)

    if DEBUG_SINGULAR:
        scov = star_covs[20]
        smn  = star_means[20]
        py_lnol = sclno(gcov, gmn, np.array([scov]), np.array([smn]), 1)
        ol.get_lnoverlaps(gcov, gmn,
                          np.array([scov]),
                          np.array([smn]), 1)

    if DEBUG_UNDERFLOW:
        scov = star_covs[0]
        smn = star_means[0]
        #py_lnol = sclno(gcov, gmn, np.array([scov]), np.array([smn]), 1)
        py_lnol = clno(gcov, gmn, scov, smn)
#        gcovi = np.linalg.inv(gcov)
#        scovi = np.linalg.inv(scov)
#        gcovi_det = np.linalg.det(gcovi)
#        scovi_det = np.linalg.det(scovi)
#        c_ol1 = ol.get_overlap(gcovi, gmn, gcovi_det,
#                                scovi, smn, scovi_det)
#        c_ol2 = ol.get_overlaps(gcovi, gmn, gcovi_det,
#                                np.array([scovi]), np.array([smn]),
#                                np.array([scovi_det]),
#                                1)
#        c_ol2 = ol.get_overlap2(gcovi.tolist(), gmn.tolist(), gcovi_det,
#                                scovi.tolist(), smn.tolist(), scovi_det)
#        logging.info("Inputs (from python to c) are:")
#        logging.debug("  A:\n{}".format(gcov))
#        logging.debug("  a:\n{}".format(gmn))
#        logging.debug("  B:\n{}".format(gcov))
#        logging.debug("  b:\n{}".format(smn))
        c_lnol1 = ol.new_get_lnoverlap(gcov, gmn,
                                       scov, smn)
#        c_lnol2 = ol.new_get_lnoverlaps(gcov, gmn,
#                                   np.array([scov]),
#                                   np.array([smn]), 1)


