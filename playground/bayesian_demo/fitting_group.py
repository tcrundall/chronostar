#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from data_generator import *
import pickle
from dynamicfitter import *
import argparse

"""
URGENT: Confirm that likelihood function is maximal for the
        bayesian fit which yielded the smallest sigma
            (or at least close to the true age!!!)
TO DO: Make Group PDF normalised such that area under cure is 1

"""

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

true_age = 8

log_filename = "bayes_fit_vel_{}.log".format(str(vel_error).replace('.','_'))

#generating stars
original_stars = generate_stars(nstars, pos_mu, pos_sig, vel_mu, vel_sig)

plt.hist(original_stars[:,0])
plt.savefig("hist_of_original_stars_pos.eps")
plt.clf()

plt.hist(original_stars[:,2])
plt.savefig("hist_of_original_stars_vel.eps")
plt.clf()

# Project stars from birth origin to "modern day"
pr_stars = project_stars(original_stars, true_age)

# "Measure" modern stars with some uncertainity
pos_error = 1 
#vel_error = 0.1
ms_stars = get_measurements(pr_stars, pos_error, vel_error)

# Trace back measured stars by projecting backwards through time
n_times = 401
max_age = 2*true_age
times = np.linspace(0, max_age, n_times)
trace_back = np.zeros((n_times, nstars, npars))
for t_ix, time in enumerate(times):
    trace_back[t_ix] = project_stars(ms_stars, time, back=True)

# Plotting summed PDF of traced back stars
# If you change xs range, will need to adjust range in plot ~ln 112
xs = np.linspace(-100,100,500)
for t_ix, time in enumerate(times):
    if t_ix%100 == 0:
        plt.plot(xs, group_pdf(xs, trace_back[t_ix]), label=time)
    elif t_ix%50 == 0:
        plt.plot(xs, group_pdf(xs, trace_back[t_ix]), '--')
plt.xlabel("X [pc]")
plt.ylabel("P(X)")
plt.title("Generalised histogram of stellar tracebacks at various ages")
plt.legend(loc='best')
filename = "generalised_histogram_vel_{}.eps".format(str(vel_error).replace('.','_'))
plt.savefig(filename)
plt.clf()

# Fitting a gaussian to summed PDF
#   vs
# Fitting a gaussian through bayesian analysis
st_fitted_sigs = np.zeros(n_times)
st_fitted_mus  = np.zeros(n_times)
ba_fitted_sigs = np.zeros(n_times)
ba_fitted_mus  = np.zeros(n_times)
group_size     = np.zeros(n_times)
init_pars = [0,100]
bnds = ((None, None), (0.1,None))
for i, time in enumerate(times):
    st_fit = opt.minimize(
        gaussian_fitter, init_pars, (nstars, trace_back[i]),
        bounds=bnds)
    ba_fit = opt.minimize(
        overlap, init_pars, (nstars, trace_back[i]), bounds=bnds)

    st_fitted_sigs[i] = np.abs(st_fit.x[1])
    st_fitted_mus[i]  = st_fit.x[0]
    ba_fitted_sigs[i] = ba_fit.x[1]
    ba_fitted_mus[i]  = ba_fit.x[0]
    group_size[i]     = get_group_size(trace_back[i])

norm_fac = 10 / np.min(group_size)
plt.plot(times, norm_fac * group_size, label="Scaled average Euclid Dist")
plt.plot(times, st_fitted_sigs, label="Standard fit")
plt.plot(times, ba_fitted_sigs, label="Bayesian fit")
plt.xlabel("Age [Myrs]")
plt.ylabel(r"Fitted $\sigma$")
plt.title("Positional variance of group fit with vel_error = {}".\
    format(vel_error))
plt.legend(loc='best')
filename = "positional_variance_vel_{}".format(str(vel_error).replace('.','_'))
plt.savefig(filename + ".eps")
plt.clf()

best_st_ix = np.argmin(st_fitted_sigs)
best_ba_ix = np.argmin(ba_fitted_sigs)
best_ed_ix = np.argmin(group_size)
st_time = times[best_st_ix]
ba_time = times[best_ba_ix]
ed_time = times[best_ed_ix]
pickle.dump((st_time, ba_time, st_fitted_sigs, ba_fitted_sigs),
            open(filename + ".pkl",'w'))

best_st_mu  = st_fitted_mus[best_st_ix]
best_ba_mu  = ba_fitted_mus[best_ba_ix]
best_ba_sig = ba_fitted_sigs[best_ba_ix]

plt.plot(xs[200:300], gaussian(xs[200:300], best_ba_mu, best_ba_sig),
         label="Bayes Fit, age= {}".format(ba_time))
plt.plot(xs[200:300], gaussian(xs[200:300], pos_mu,     pos_sig),     label="Original PDF")
plt.legend(loc='best')
plt.xlabel("X [pc]")
plt.ylabel("P(X)")
plt.title("PDFs of Original Group and the Bayesian Fit")
filename = "pdf_orig_vs_bayes_vel_{}.eps".format(str(vel_error).replace('.','_'))
plt.savefig(filename)
plt.clf()

log_filename = "bayes_fit_vel_{}.log".format(str(vel_error).replace('.','_'))
with open(log_filename, 'w') as f:
    f.write("With {} stars and velocity measurement error of {} ... \n"\
                .format(nstars, vel_error))
    f.write("True age:  {}\n".format(true_age))
    f.write("True position mean : {}\n".format(pos_mu))
    f.write("True position sigma: {}\n".format(pos_sig))
    f.write("True velocity mean : {}\n".format(vel_mu))
    f.write("True velocity sigma: {}\n".format(vel_sig))
    f.write("\n")

    f.write("__ Euclidean Distance fit __\n")
    f.write("Fitted age: {}\n".format(ed_time))
    f.write("\n")

    f.write("__ Gaussian fit __\n")
    f.write("Fitted age: {}\n".format(st_time))
    f.write("Fitted position mean: {:.2f}\n".format(best_st_mu))
    f.write("\n")

    f.write("__ Bayes fit __\n")
    f.write("Fitted age: {}\n".format(ba_time))
    f.write("Fitted position mean: {:.2f}\n".format(best_ba_mu))
    f.write("Fitted position sigma: {:.2f}\n".format(best_ba_sig))
    f.write("\n")

#plt.hist(original_stars[:,0])
#plt.show()
