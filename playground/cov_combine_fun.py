from __future__ import print_function, division

import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.converter as cv

def convertListToCovMatrix(data):
    ra = data[0]
    e_ra = data[1] / 3600. / 1000.
    dec = data[2]
    e_dec = data[3] / 3600. / 1000.
    plx = data[4]
    e_plx = data[5]
    pmra = data[6]
    e_pmra = data[7]
    pmdec = data[8]
    e_pmdec = data[9]
    rv = data[10]
    e_rv = data[11]
    c_ra_dec = data[12]
    c_ra_plx = data[13]
    c_ra_pmra = data[14]
    c_ra_pmdec = data[15]
    c_dec_plx = data[16]
    c_dec_pmra = data[17]
    c_dec_pmdec = data[18]
    c_plx_pmra = data[19]
    c_plx_pmdec = data[20]
    c_pmra_pmdec = data[21]

    mean = np.array((ra, dec, plx, pmra, pmdec, rv))
    cov  = np.array([
        [e_ra**2, c_ra_dec*e_ra*e_dec, c_ra_plx*e_ra*e_plx,
            c_ra_pmra*e_ra*e_pmra, c_ra_pmdec*e_ra*e_pmdec, 0.],
        [c_ra_dec*e_ra*e_dec, e_dec**2, c_dec_plx*e_dec*e_plx,
            c_dec_pmra*e_dec*e_pmra, c_dec_pmdec*e_dec*e_pmdec, 0.],
        [c_ra_plx*e_ra*e_plx, c_dec_plx*e_dec*e_plx, e_plx**2,
            c_plx_pmra*e_plx*e_pmra, c_plx_pmdec*e_plx*e_pmdec, 0.],
        [c_ra_pmra*e_ra*e_pmra, c_dec_pmra*e_dec*e_pmra,
                                                c_plx_pmra*e_plx*e_pmra,
             e_pmra**2, c_pmra_pmdec*e_pmra*e_pmdec, 0.],
        [c_ra_pmdec*e_ra*e_pmdec, c_dec_pmdec*e_dec*e_pmdec,
                                                c_plx_pmdec*e_plx*e_pmdec,
             c_pmra_pmdec*e_pmra*e_pmdec, e_pmdec**2, 0.],
        [0., 0., 0., 0., 0., e_rv**2]
    ])
    return mean, cov


star1 = [6.959867718, 0.062806964, -32.55195499, 0.03829661, 28.65910289, 0.07249356, 109.8165403, 0.114603975, -47.38920138, 0.067204367, 8.8, 0.2, -0.05326207, 0.433427, 0.19840564, -0.38549396, 0.16367765, -0.321486, -0.5215564, 0.062251348, -0.28380314, -0.1701706]
star2 = [6.960380814, 0.06589466, -32.55689462, 0.040584455, 28.68067198, 0.076973018, 112.2114753, 0.116485588, -44.6378334, 0.071023096, 8.5, 0.5, -0.048581935, 0.46184567, 0.21392117, -0.39568934, 0.17742275, -0.31782857, -0.54934806, 0.06892159, -0.30934203, -0.17179945]

mean1, cov1 = convertListToCovMatrix(star1)
mean2, cov2 = convertListToCovMatrix(star2)

av_cov = (cov1 + cov2) / 2
av_star = (np.array(star1) + np.array(star2)) * 0.5
av_cov_from_av_star = convertListToCovMatrix(av_star)
