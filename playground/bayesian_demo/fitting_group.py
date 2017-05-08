#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from data_generator import *
import pickle
from dynamicfitter import *
import argparse

"""
USAGE:  run with `./fitting_group.py -v [velocity measurement error]`
        Will finish with a pdb.set_trace() giving you access to np arrays with
            the stars and their overlaps for two instances:
                - best fit: the fit which yielded largest likelihood
                - narrowest fit: the fit which was narrowest
        Note that time isn't a free paramter in this code, rather I just
            iterate through each time index. The mid index (40 at the moment)
            is the "true" age

        Will generate a log:
            bayes_fit_vel_[velocity error].eps

        Also a pkl file:
            overlap_vel_[velocity error].pkl
                - a bit useless to be honest. Feel free to change what's put
                    in here. There aren't any dependencies to worry about

        Will generate some plots:
            hist_of_original_stars_pos.eps
            hist_of_original_stars_vel.eps
            generalised_histogram_vel_[velocity error].eps
                - gen-hist of traceback position at various times
            positional_variance_vel_[velocity error].eps
                - plot showing how well each approach matches the correct age
                - !!! use this to confirm the fitting is smooth, i.e. no 
                    random jumps/spikes (which happens when optimiser gets
                    confused)
            overlap_vel_[velocity error].eps
                - shows the raw likelihood result (lnlike) along with the 
                   adjusted result with a heavy prior on width of group (lnpost)
            bayes_fit_vel_[velocity error].eps
                - compares the original position distribution of stars along
                    with the narrowest bayesian fitted gaussian
            

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
nstars = 100
pos_mu  = 0
pos_sig = 5
vel_mu  = 0
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
n_times = 81
max_age = 2*true_age
times = np.linspace(0, max_age, n_times)
trace_back = np.zeros((n_times, nstars, npars))
for t_ix, time in enumerate(times):
    trace_back[t_ix] = project_stars(ms_stars, time, back=True)

# Plotting summed PDF of traced back stars
# If you change xs range, will need to adjust range in plot ~ln 112
xs = np.linspace(-100,100,500)
for t_ix, time in enumerate(times):
    if t_ix%20 == 0:
        plt.plot(xs, group_pdf(xs, trace_back[t_ix]), label=time)
    elif t_ix%10 == 0:
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
st_fitted_sigs = np.zeros(n_times) # fitting gaussian to generaised histogram
st_fitted_mus  = np.zeros(n_times) # ^^ ditto
ba_fitted_sigs = np.zeros(n_times) # gausssian pars that maximise likelihood
ba_fitted_mus  = np.zeros(n_times) # ^^ ditto
group_size     = np.zeros(n_times) # euclid distance
overlaps       = np.zeros(n_times) # includes strong prior on stdev of pos
raw_overlaps   = np.zeros(n_times) # raw overlap integral at each time step
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
    group_size[i]     = get_group_size(trace_back[i]) #euclid distance
    raw_overlaps[i]   = overlap(ba_fit.x, nstars, trace_back[i])
    overlaps[i]       = lnprior(ba_fit.x[1]) - raw_overlaps[i]

#pdb.set_trace()
norm_fac = 10 / np.min(group_size)
plt.plot(times, norm_fac * group_size, label="Scaled average Euclid Dist")
plt.plot(times, st_fitted_sigs, label="Standard fit")
plt.plot(times, ba_fitted_sigs, label="Bayesian fit")
plt.plot(times, 10*overlaps/np.min(overlaps),         label="lnPost")
plt.plot(times,  5*raw_overlaps/np.max(raw_overlaps), label="lnLike")
plt.xlabel("Age [Myrs]")
plt.ylabel(r"Fitted $\sigma$")
plt.title("Positional variance of group fit with vel_error = {}".\
    format(vel_error))
plt.legend(loc='best')
filename = "positional_variance_vel_{}".format(str(vel_error).replace('.','_'))
plt.savefig(filename + ".eps")
plt.clf()

plt.plot(times, overlaps,      label="lnPost")
plt.plot(times, -raw_overlaps, label="lnLike")
plt.legend(loc='best')
filename = "overlap_vel_{}".format(str(vel_error).replace('.','_'))
plt.savefig(filename + ".eps")
plt.clf()

best_st_ix = np.argmin(st_fitted_sigs)
#best_ba_ix = np.argmin(ba_fitted_sigs)
best_ba_ix = np.argmax(overlaps) # use the fit with largest lnlike
best_ed_ix = np.argmin(group_size)
smallest_ba_ix = np.argmin(ba_fitted_sigs)
st_time = times[best_st_ix]
ba_time = times[best_ba_ix]
ed_time = times[best_ed_ix]
smallest_ba_time = times[smallest_ba_ix]

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

# For debugging reasons, extract the traceback positions for the 
#  time which yielded the best (highest lnprob) bayesian fit and
#  the time which yielded the narrowest bayesian fit

# Gathering bayesian fitted parameters into groups
group_pars = np.stack((ba_fitted_mus, ba_fitted_sigs), axis=1)

best_fit_stars = trace_back[best_ba_ix][:][:]
best_fit_group = group_pars[best_ba_ix]

narrowest_fit_stars = trace_back[smallest_ba_ix][:][:]
narrowest_fit_group = group_pars[smallest_ba_ix]

best_fit_overlaps = np.zeros(nstars)
narrowest_fit_overlaps = np.zeros(nstars)

for i in range(nstars):
    best_fit_overlaps = single_overlap(
        best_fit_group, best_fit_stars[i][0:2]
        )
    narrowest_fit_overlaps = single_overlap(
        narrowest_fit_group, narrowest_fit_stars[i][0:2]
        )
# Now you can.. i dunno, plot a histogram of overlaps... mimic the for loops
#  from above to plot positions of stars...

pdb.set_trace()
