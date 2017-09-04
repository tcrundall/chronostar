#! /usr/bin/env python
"""
Investigate the reliability of age fitting.

Generates a varied set of synthetic astrometry datasets.
Run traceback and fit, compare fitted age with ground truth age.
"""
from chronostar import synthesiser as syn
import numpy as np
from subprocess import call
from chronostar.fit_groups import fit_groups
import pdb
import matplotlib.pyplot as plt

group_pars = np.array([
    # X, Y, Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,N
    [ 0, 0, 0,0,0,0,10,10,10, 3,  0,  0,  0,20,30],
    [ 0,10, 0,0,0,0, 5, 5, 5, 3,0.6,0.5,0.0,20,40],
    [10, 0, 0,0,0,0,10,10,10, 3,0.4,0.2,0.0,20,50],
    [10,10, 0,0,0,0, 5, 5, 5, 3,  0,0.2,0.0,10,60],
    [10,10,10,0,0,0,10,10,10, 3,  0,0.2,0.0,10,20],
    ])
# error = 0.01
ntests = group_pars.shape[0]
true_ages = []
fitted_ages = []

base_group = [ 0,10, 0,0,0,0, 5, 5, 5, 3,0.6,0.5,0.0]
for age in [5,7.5,10,12.5,15,17.5,20,22.5,25,27.5,30]:
    for nstars in [15, 30, 60]:
        group_pars = base_group + [age] + [nstars]
        syn.synthesise_data(1, group_pars, 0.01)
        astr_filename =\
            "data/synth_data_1groups_{}stars.pkl".format(nstars)
        tb_filename =\
            "data/tb_synth_data_1groups_{}stars.pkl".format(nstars)
        call([
            "python", "generate_default_tb.py",
            astr_filename,
            ])
        best_fit = fit_groups(
            1000,1000,1,0,tb_filename,tstamp=nstars,noplots=True)

        true_ages.append(age)
        fitted_ages.append(best_fit[-2])
    
plt.plot(true_ages[0::3], fitted_ages[0::3], "bo", label="15 stars")
plt.plot(true_ages[1::3], fitted_ages[1::3], "g+", label="30 stars")
plt.plot(true_ages[2::3], fitted_ages[2::3], "yx", label="60 stars")
plt.legend(loc='best'); plt.xlabel("True age"); plt.ylabel("Fitted age")

pdb.set_trace()

# synthesise a bunch of solo groups with varying error
# errors = [0.01, 0.02, 0.05, 0.1]

# for error in errors:

# # synthesise a bunch of solo groups
# for i in range(5):
#     syn.synthesise_data(1, group_pars[i], error)
# 
# # synthesise a bunch of paired groups
# for i in range(4):
#     syn.synthesise_data(2, group_pars[i:i+2], error)
# 
# # synthesise a couple of three groups
# syn.synthesise_data(3, group_pars[0:3], error)
# syn.synthesise_data(3, group_pars[2:5], error)

