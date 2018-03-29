#! usr/bin/env python

import numpy as np
import pickle
import sys
sys.path.insert(0, '..')

import chronostar.tfgroupfitter as tfgf
import chronostar.synthesiser as syn
import chronostar.traceback as tb

dX = 10
dV = 5
age = 10
nstars = 100
group_pars_old_style = [0,0,0,0,0,0,dX,dX,dX,dV,0.,0.,0., age, nstars]
group_pars_tf_style = [0,0,0,0,0,0,dX,dV,age]

tb_file = 'temp_traceforward_tb.pkl'
synth_file = 'temp_traceforward_synth.pkl'
error = 1e-5
try:
    print("Trying to open file")
    with open(tb_file):
        pass
# if not created, then create it. Careful though! May not be the same
# as group_pars. So if test fails try deleting tb_file from
# directory
except IOError:
    print("Generating new file")
    # generate synthetic data
    syn.synthesise_data(
        1, group_pars_old_style, error, savefile=synth_file
    )
    with open(synth_file, 'r') as fp:
        t = pickle.load(fp)

    times = np.linspace(0, 1, 2)
    tb.traceback(t, times, savefile=tb_file)

group_pars_in = np.copy(group_pars_tf_style)
group_pars_in[6:8] = 1 / group_pars_in[6:8]

# find best fit
burnin_steps = 400
sampling_steps = 1000
print("Burning in: {}".format(burnin_steps))
print("Sampling: {}".format(sampling_steps))
best_fit, sampler_chain, lnprob = tfgf.fit_group(
    tb_file, burnin_steps=burnin_steps, sampling_steps=sampling_steps,
    plot_it=True
)
means = best_fit[0:6]
stds = 1 / best_fit[6:8]
age = best_fit[8]

tol_mean = 3.5
tol_std = 2.5
tol_age = 0.5

assert np.max(abs(means - group_pars_tf_style[0:6])) < tol_mean,\
   "\nReceived:\n{}\nshould be within {} to:\n{}".\
        format(means, tol_mean, group_pars_tf_style[0:6])
assert np.max(abs(stds - group_pars_tf_style[6:8])) < tol_std,\
    "\nReceived:\n{}\nshould be close to:\n{}".\
        format(stds, tol_std, group_pars_tf_style[6:8])
assert np.max(abs(age - group_pars_tf_style[8])) < tol_age,\
    "\nReceived:\n{}\nshould be close to:\n{}".\
    format(age, tol_age, group_pars_tf_style[8])


