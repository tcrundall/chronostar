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
import pickle

group_pars = np.array([
    # X, Y, Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,N
    [ 0, 0, 0,0,0,0,10,10,10, 3,  0,  0,  0,20,30],
    [ 0,10, 0,0,0,0, 5, 5, 5, 3,0.6,0.5,0.0,20,40],
    [10, 0, 0,0,0,0,10,10,10, 3,0.4,0.2,0.0,20,50],
    [10,10, 0,0,0,0, 5, 5, 5, 3,  0,0.2,0.0,10,60],
    [10,10,10,0,0,0,10,10,10, 3,  0,0.2,0.0,10,20],
    ])
# error = 0.01
# ntests = group_pars.shape[0]

nburnin = 1000
nsteps = 1000

base_group = [ 0,10, 0,0,0,0, 5, 5, 5, 3,0.6,0.5,0.0]
for error in [1.2,1.4,1.6,1.8,2.0]: #[0.01,0.1,0.2,0.4,0.6,0.8,1.0]:
    true_ages = []
    fitted_ages = []
    print("_____ Running for error {} _____".format(error))
    for age in [5,10,15,20,25,30]:
        print("  ___ Fitting age {} Myr ___".format(age))
        for nstars in [15,30,60]:
            group_pars = base_group + [age] + [nstars]
            syn.synthesise_data(1, group_pars, error)
            astr_filename =\
                "data/synth_data_1groups_{}stars.pkl".format(nstars)
            tb_filename =\
                "data/tb_synth_data_1groups_{}stars.pkl".format(nstars)
            call([
                "python", "generate_default_tb.py",
                astr_filename,
                ])

            info = "Initial conditions: {}\n".format(group_pars) +\
                   "Error: {} %\n".format(error) +\
                   "nburnin, nsteps: {}, {}\n".format(nburnin, nsteps)

            best_sample, best_fits = fit_groups(
                nburnin,nsteps,1,0,tb_filename,tstamp=nstars,noplots=True,
                info=info)

            true_ages.append(age)
            fitted_ages.append(best_fits[-2])
        
    fitted_ages = np.array(fitted_ages)
    fitted_ages[0::3][1] = 0
    plt.clf()
    plt.plot(true_ages[0::3], fitted_ages[0::3,0], "bo", label="15 stars")
    plt.errorbar(
        true_ages[0::3], fitted_ages[0::3,0], fmt="none", capsize=5,ecolor='b',
        yerr=[fitted_ages[0::3,2], fitted_ages[0::3,1]])
    plt.plot(true_ages[1::3], fitted_ages[1::3,0], "rs", label="30 stars")
    plt.errorbar(
        true_ages[1::3], fitted_ages[1::3,0], fmt="none", capsize=5,ecolor='r',
        yerr=[fitted_ages[1::3,2], fitted_ages[1::3,1]])
    plt.plot(true_ages[2::3], fitted_ages[2::3,0], "y^", label="60 stars")
    plt.errorbar(
        true_ages[2::3], fitted_ages[2::3,0], fmt="none", capsize=5,ecolor='y',
        yerr=[fitted_ages[2::3,2], fitted_ages[2::3,1]])
    plt.plot([0,35],[0,35],'g-', label='target')
    #plt.plot(true_ages, fitted_ages, "bo", label="30 stars")
    plt.title("Error = {}%".format(int(error*100)))
    plt.legend(loc='best'); plt.xlabel("True age"); plt.ylabel("Fitted age")
    filename = "age_diagnostic_{}_error".format(int(error*100))
    plt.savefig("plots/"+filename+".eps")
    pickle.dump((true_ages, fitted_ages), open("results/"+filename+'.pkl', 'w'))
    
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

