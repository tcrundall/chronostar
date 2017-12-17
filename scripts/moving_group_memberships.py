#!/usr/bin/env python

import sys
sys.path.insert(0,'..')

import numpy as np
from scipy.stats import gmean
#import chronostar.groupfitter as gf
import pickle
from chronostar.utils import generate_cov
from chronostar._overlap import get_lnoverlaps
import pdb

def group_pars_from_malo(malo_pars):
    """Convert malo+ 2013 moving group pars into chronostar format

    TODO:
        confirm if X needs to be negafied...

    Parameters
    ----------
    malo_pars : [12] array
        U, V, W, dU, dV, dZ, X, Y, Z, dX, dY, dZ

    Returns
    -------
    group_pars : [14] array
        [X, Y, Z, U, V, W, 1/dX, 1/dY, 1/dZ, 1/dV, Cxy, Cxz, Cyz, age]
    """
    group_pars = np.zeros(14)
    group_pars[0]   = -malo_pars[6]
    group_pars[1:3] =  malo_pars[7:9]
    group_pars[3:6] =  malo_pars[0:3]
    group_pars[6:9] =  1/malo_pars[9:12]
    group_pars[9]   =  1/gmean(malo_pars[3:6])
    return group_pars

def test_pars_swap():
    malo_pars = np.array([
        -10.94, -16.25,  -9.27,  2.06,  1.30,  1.54,   9.27,  -5.96, -13.59, 31.71, 15.19,  8.22
    ])
    dV = gmean(malo_pars[3:6])
    chro_pars = np.array([
        -9.27,  -5.96, -13.59, -10.94, -16.25,  -9.27, 1/31.71, 1/15.19,  1/8.22, 1/dV,
        0.0, 0.0, 0.0, 0.0
    ])
    assert np.allclose(chro_pars, group_pars_from_malo(malo_pars))

def calc_lnoverlaps(group_pars, stars_mns, stars_covs, nstars):
    """Crappy wrapper to find overlap integrals from pars encoding

    """
    group_mn = group_pars[0:6]
    # convert pars into (in?)covariance matrix
    group_cov = generate_cov(group_pars)
    # check if covariance matrix is singular
    assert np.min( np.linalg.eigvalsh(group_cov) ) >= 0

    # interpolate star data to modelled age
    age = group_pars[13]

    lnols = get_lnoverlaps(
        group_cov, group_mn, stars_covs, stars_mns, nstars
    )
    return lnols

def calc_membership_probs(star_lnols):
    """Calculate probabilities of membership from log overlaps
    """
    ngroups = star_lnols.shape[0]
    star_memb_probs = np.zeros(ngroups)
    
    for i in range(ngroups):
        star_memb_probs[i] = 1./np.sum(np.exp(star_lnols - star_lnols[i]))

    return star_memb_probs 

def test_calc_memb():
    star_lnols = np.array([-100, -100, -100, -100, -100])
    ans = [0.2, 0.2, 0.2, 0.2, 0.2]
    assert np.allclose(calc_membership_probs(star_lnols), ans)

test_pars_swap()
test_calc_memb()

table_1 = np.array([
    [-10.1, 2.1, -15.9, 0.8,  -9.2, 1.0, 48],
    [ -9.9, 1.5, -20.9, 0.8,  -1.4, 0.9, 44], 
    [-13.2, 1.3, -21.8, 0.8,  -5.9, 1.2, 41],
    [-10.2, 0.4, -23.0, 0.8,  -4.4, 1.5, 23],
    [-10.5, 0.9, -18.0, 1.5,  -4.9, 0.9, 22],
    [-11.0, 1.2, -19.9, 1.2, -10.4, 1.6, 24],
    [-14.5, 0.9,  -3.6, 1.6, -11.2, 1.4, 15],
    [-22.0, 0.3, -14.4, 1.3,  -5.0, 1.3, 64],
    [ -6.8, 1.3, -27.2, 1.2, -13.3, 1.6, 89],
])

table_2 = np.array([
    [ 20,  -36,  76,   -5,  -33,  21, -15, -29,  -1,  31, 21, 10],
    [  3,  -61,  43,  -24,  -47,  -4, -35, -44, -30,  48,  7, 30],
    [-42, -106,   9,  -56, -168,   1, -47, -99,   6,  82, 30, 30],
    [ 14,   -2,  33,  -94, -154, -39, -17, -33,   5,  85, 35, 30],
    [ 15,    2,  34,  -44,  -61, -26,  21,  10,  27,  48, 13,  8],
    [ 50,   34,  60,  -92, -105, -78, -28, -44, -12, 108,  9,  6],
    [ 22,  -79, 142, -106, -138, -60, -68, -85, -38, 141, 34, 20],
    [  5,  -55,  64, -115, -154,  -6, -18, -67,   8, 106, 51, 40],
    [ -6,  -94,  73,  -14, -131,  58, -20, -66,  23,  34, 26, 70],
])

malo_cols = ['U', 'V', 'W', 'dU', 'dV', 'dZ', 'X', 'Y', 'Z', 'dX', 'dY', 'dZ']
malo_rows = ['bpmg', 'tha', 'abdmg', 'col', 'car', 'twa', 'arg', 'field']

malo_table_2 = np.array([
    [-10.94, -16.25,  -9.27,  2.06,  1.30,  1.54,   9.27,  -5.96, -13.59, 31.71, 15.19,  8.22],
    [ -9.88, -20.70,  -0.90,  1.51,  1.87,  1.31,  11.39, -21.21, -35.40, 19.29,  9.17,  5.39],
    [ -7.12, -27.31, -13.81,  1.39,  1.31,  2.16,  -2.37,   1.48, -15.62, 20.03, 18.83, 16.59],
    [-12.24, -21.32,  -5.58,  1.03,  1.18,  0.89, -27.44, -31.32, -27.97, 13.79, 20.55, 15.09],
    [-10.50, -22.36,  -5.84,  0.99,  0.55,  0.14,  15.55, -58.53, -22.95,  5.66, 16.69,  2.74],
    [ -9.87, -18.06,  -4.52,  4.15,  1.44,  2.80,  12.49, -42.28,  21.55,  7.08,  7.33,  4.20],
    [-21.78, -12.08,  -4.52,  1.32,  1.97,  0.50,  14.60, -24.67,  -6.72, 18.60, 19.06, 11.43],
    [-10.92, -13.35,  -6.79, 23.22, 13.44,  8.97,  -0.18,   2.10,   3.27, 53.29, 51.29, 50.70],
])

field_pars = np.array([
        -3.23224760e+01,  -5.20039277e+01,  -2.79503162e+01,
         1.47052647e+00,  -6.96123777e+00,   1.05548836e-01,
         1.40728422e-02,   1.72754663e-02,   1.01462923e-02,
         4.14475921e-02,  -2.71573687e-01,  -2.09653483e-01,
        -2.83523985e-01,   0.00000000e+00
])

tb_file = '../data/tb_rave_active_star_candidates_with_TGAS_kinematics.pkl'
with open(tb_file, 'r') as fp:
    t, _, xyzuvw, xyzuvw_cov = pickle.load(fp)

nstars = xyzuvw.shape[0]
ngroups = len(malo_rows)
lnols = np.zeros((nstars, len(malo_rows)))

# CALCULATE OVERLAPS
# disregarding malo 'field' group cause I have a better field fit
for i, (group_name, malo_pars) in enumerate(zip(malo_rows[:-1], malo_table_2[:-1])):
    print(i)
    print(group_name)
    print(malo_pars)
    chro_pars = group_pars_from_malo(malo_pars)
    lnols[:,i] = calc_lnoverlaps(chro_pars, xyzuvw[:,0], xyzuvw_cov[:,0], nstars)

lnols[:,-1] = calc_lnoverlaps(field_pars, xyzuvw[:,0], xyzuvw_cov[:,0], nstars)

# CONVERT OVERLAPS TO MEMBERSHIP PROBABILITIES
#memb_probs = np.zeros( (nstars, ngroups) )
memb_probs = {}
for i in range(nstars):
    memb_probs[malo_rows[i] = calc_membership_probs(lnols[i])
    #memb_probs[i] = calc_membership_probs(lnols[i])

# CALC 10 MOST LIKELY MEMBERS OF EACH GROUP
best_membs_ixs = {}
nbest = 10
#best_membs_ixs = np.zeros((ngroups,nbest), dtype=np.int8)
perc = 100 * (1 - (1.*nbest)/nstars)
for i in range(ngroups):
    #best_membs_ixs[i] = np.where(memb_probs[:,i] > np.percentile(memb_probs[:,i], perc))[0]
    best_membs_ixs[malo_rows[i]] =\
        np.where(memb_probs[:,i] > np.percentile(memb_probs[:,i], perc))[0]
#    best_membs_ixs.append(
#        np.where(memb_probs[:,i] > np.percentile(memb_probs[:,i], perc))[0]

# SANITY CHECK, see if means and stds of most likely members are
# somewhat consistent with malo group pars

bpmg_ixs = best_membs_ixs['bpmg']
retro_bpmg_pars = np.mean(xyzuvw[bpmg_ixs,0], axis=0)

pdb.set_trace()
