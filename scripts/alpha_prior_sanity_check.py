"""
This script explores the impact the prior on the virial should have on
permitting low density components mapping to the field.
"""
from __future__ import print_function, division

import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.synthesiser as syn
import chronostar.groupfitter as gf

# 1 comp fit
one_group_pars_ex = [
    np.array([27.21857533, 40.84870565, 23.3006967 , -0.96968654, -3.47371966,
              -0.29458631, 16.70523603,  1.15738955, 14.07591161])
]
one_weights = [61.9987919]

# 2 comp fit
two_group_pars_ex = [
    np.array([25.24100057, 35.37151966, 23.73419406,  0.68106934, -3.77992686,
              -0.43692936,  8.97195291,  0.92658985, 13.33282707]),
    np.array([ 2.97783565e+01,  4.85637518e+01,  2.45364372e+01, -2.58377259e+00,
              -3.04098661e+00, -3.34909874e-02,  2.16584223e+01,  8.31520834e-01,
               1.53740845e+01])
]
two_weights = [33.02902374, 28.97088176,  6.0000945]

# 3 comp fit
three_group_pars_ex = [
    np.array([26.54427126, 37.99652339, 24.08375344,  0.36762423, -3.7307971,
              -0.34351942,  9.02614369,  0.92211963, 14.14381967]),
    np.array([  8.93722359,  22.28633376,  -1.18878485,  -8.63887102,
              -11.130962,  -3.43125315,   8.51810544,   1.77553,
                1.51311008]),
    np.array([ 3.21488735e+01,  4.88207224e+01,  2.43963555e+01, -2.75347262e+00,
              -2.98063139e+00,  1.26400066e-02,  2.21896254e+01,  7.69173481e-01,
               1.60651391e+01])
]
three_weights = [36.60656637,  4.70398271, 24.68926151,  2.00018941]

# gather everything up
all_group_pars = [one_group_pars_ex,
                  two_group_pars_ex,
                  three_group_pars_ex]
all_weights = [one_weights,
               two_weights,
               three_weights]

# For each fit
for group_pars_ex, weights in zip(all_group_pars, all_weights):
    lnalpha_priors = []

    for group_par, weight in zip(group_pars_ex, weights):
        group_obj = syn.Group(group_par, starcount=False, internal=False,
                              sphere=True)
        lnalpha_priors.append(gf.lnAlphaPrior(group_obj.getInternalSphericalPars(),
                                              None, weight))
    print("{} component fit".format(len(group_pars_ex)))
    print(lnalpha_priors)
