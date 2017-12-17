#!/usr/bin/env python

import sys
sys.path.insert(0, '..')

import chronostar.groupfitter as gf
import pickle
import matplotlib.pyplot as plt
import numpy as np


print("New1")
infile = "../data/tb_rave_active_star_candidates_with_TGAS_kinematics.pkl"

with open(infile, 'r') as fp:
    stars, ts, xyzuvw, xyzuvw_cov = pickle.load(fp)

#init_pars = np.zeros(14)
#init_pars[0:6] = np.mean(xyzuvw[:,0,:], axis=0)
#init_pars[6:10] = 1/np.std(xyzuvw[:,0,:4], axis=0)
#
#init_pars = np.array([
#        -3.25428874e+01,  -5.57343789e+01,  -3.26822860e+01,
#         1.45180826e+00,  -6.50189696e+00,   3.05571179e-01,
#         1.38614076e-02,   1.72206362e-02,   1.01477181e-02,
#         4.11418947e-02,  -2.85696722e-01,  -2.15276243e-01,
#        -2.91917283e-01,   0.00000000e+00
#])
#
init_pars = np.array([
        -3.23224760e+01,  -5.20039277e+01,  -2.79503162e+01,
         1.47052647e+00,  -6.96123777e+00,   1.05548836e-01,
         1.40728422e-02,   1.72754663e-02,   1.01462923e-02,
         4.14475921e-02,  -2.71573687e-01,  -2.09653483e-01,
        -2.83523985e-01,   0.00000000e+00
])

print(init_pars)

best_fit, chain = gf.fit_group(
    infile, plot_it=True, fixed_age=0.0, init_pars=init_pars
)

with open("fit_result.pkl", 'w') as fp:
    pickle.dump(best_fit, fp)

#pdb.set_trace()
