#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from data_generator import *
import pickle
from dynamicfitter import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-v', dest='v', default=1)

args = parser.parse_args()
vel_error = float(args.v)

# Setting up true group parameters
nstars = 50
pos_mu  = 0
pos_sig = 5
vel_mu  = 2
vel_sig = 5
npars   = 4 #number of parameters required to describe a star

true_age = 20

#generating stars
original_stars = generate_stars(nstars, pos_mu, pos_sig, vel_mu, vel_sig)

# Project stars from birth origin to "modern day"
pr_stars = project_stars(original_stars, true_age)

# "Measure" modern stars with some uncertainity
pos_error = 1 
#vel_error = 0.1
ms_stars = get_measurements(pr_stars, pos_error, vel_error)

# Trace back measured stars by projecting backwards through time
n_times = 201
max_age = 2*true_age
times = np.linspace(0, max_age, n_times)
trace_back = np.zeros((n_times, nstars, npars))
for t_ix, time in enumerate(times):
    trace_back[t_ix] = project_stars(ms_stars, time, back=True)

# Plotting summed PDF of traced back stars
xs = np.linspace(-100,100,500)
for t_ix, time in enumerate(times):
    plt.plot(xs, group_pdf(xs, trace_back[t_ix]), label=time)
#plt.show()
plt.clf()

# Fitting a gaussian to summed PDF
#   vs
# Fitting a gaussian through bayesian analysis
st_fitted_sigs = np.zeros(n_times)
ba_fitted_sigs  = np.zeros(n_times)
ba_fitted_mus   = np.zeros(n_times)
init_pars = [0,100]
bnds = ((None, None), (0.1,None))
for i, time in enumerate(times):
    st_fit = opt.minimize(
        gaussian_fitter, init_pars, (nstars, trace_back[i]),
        bounds=bnds)
    ba_fit = opt.minimize(
        overlap, init_pars, (nstars, trace_back[i]), bounds=bnds)

    st_fitted_sigs[i] = np.abs(st_fit.x[1])
    ba_fitted_sigs[i] = ba_fit.x[1]
    ba_fitted_mus[i]  = ba_fit.x[0]

plt.plot(times, st_fitted_sigs, label="Standard fit")
plt.plot(times, ba_fitted_sigs, label="Bayesian fit")
plt.xlabel("Age [Myrs]")
plt.ylabel(r"Fitted $\sigma$")
plt.title("Positional variance of group fit with vel_error = {}".\
    format(vel_error))
plt.legend(loc='best')
filename = "positional_variance_vel_{}".format(str(vel_error))
plt.savefig(filename + ".eps")
plt.clf()

best_st_ix = np.argmin(st_fitted_sigs)
best_ba_ix = np.argmin(ba_fitted_sigs)
st_time = times[best_st_ix]
ba_time = times[best_ba_ix]
pickle.dump((st_time, ba_time, st_fitted_sigs, ba_fitted_sigs),
            open(filename + ".pkl",'w'))

best_ba_mu  = ba_fitted_mus[best_ba_ix]
best_ba_sig = ba_fitted_sigs[best_ba_ix]

plt.plot(xs, gaussian(xs, best_ba_mu, best_ba_sig), label="Bayes Fit")
plt.plot(xs, gaussian(xs, pos_mu,     pos_sig),     label="Original PDF")
plt.legend(loc='best')
plt.show()

plt.hist(original_stars[:,0])
plt.show()

pdb.set_trace()
