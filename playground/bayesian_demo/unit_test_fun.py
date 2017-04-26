#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from fun import *
import pickle
from dynamicfitter import *

xs = np.linspace(-10,10,21)
assert(xs[np.argmax(gaussian(xs,0,1))] == 0.0)

star0 = [1, 1, 1, 1]
time = 1
for time in range(10):
    assert(project_star(star0, time)[0] == star0[0] + star0[2]*time)
    assert(project_star(star0, time)[1] == star0[1] + star0[3]*time)

trace_back, n_time_steps, nstars, times, orig =\
    pickle.load(open("data.pkl", 'r'))

assert np.shape(trace_back) == (n_time_steps, nstars, 4)

assert gaussian_fitter((0,2), nstars, trace_back[-1]) > 0

init_pars = [0,1]
bnds = ((None, None), (0.1,None))

standard_fit = opt.minimize(
    gaussian_fitter, init_pars, (nstars, trace_back[-1]),
    bounds=bnds )
bayesian_fit = opt.minimize(overlap, init_pars, (nstars, trace_back[-1]),
    bounds=bnds )

st_fitted_mu  = standard_fit.x[0]
st_fitted_sig = standard_fit.x[1]

b_fitted_mu  = bayesian_fit.x[0]
b_fitted_sig = bayesian_fit.x[1]

xs = np.linspace(-100,100,1000)
plt.plot(xs, nstars * gaussian(xs, st_fitted_mu, st_fitted_sig), label="Gaussian fit")
plt.plot(xs, nstars * gaussian(xs, b_fitted_mu,  b_fitted_sig), label="Bayesian fit")
plt.plot(xs, group_pdf(xs, orig), label="True origin")
plt.legend(loc='best')
#plt.show()
plt.clf()

st_fitted_sigs = np.zeros(n_time_steps)
ba_fitted_sigs  = np.zeros(n_time_steps)
ba_fitted_mus   = np.zeros(n_time_steps)
init_pars = [0,2]
for i, time in enumerate(times):
    st_fit = opt.minimize(gaussian_fitter, init_pars, (nstars, trace_back[i]))
    ba_fit = opt.minimize(overlap, init_pars, (nstars, trace_back[i]))

    st_fitted_sigs[i] = np.abs(st_fit.x[1])
    ba_fitted_sigs[i] = ba_fit.x[1]
    ba_fitted_mus[i]  = ba_fit.x[0]

plt.plot(st_fitted_sigs, label="Standard fit")
plt.plot(ba_fitted_sigs, label="Bayesian fit")
plt.xlabel("Age [Myrs]")
plt.ylabel(r"Fitted $\sigma$")
plt.title("Positional variance of group fit")
plt.legend(loc='best')
plt.show()

init_skip = 2

best_ix = np.argmin(ba_fitted_sigs[init_skip:40]) + init_skip
print(times[best_ix])
print(ba_fitted_sigs[best_ix])
print(ba_fitted_mus[best_ix])
pdb.set_trace()
