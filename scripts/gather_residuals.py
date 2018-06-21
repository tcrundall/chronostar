from __future__ import division, print_function
"""
This script goes through a set of synthetic fit results and, using the 
final chains of the sampling along with the true data initialisation
parameters, calculates the residuals (how much I'm off by) and the
normalised residuals (if how much I think I'm off by is consistent with
how much I'm off by)
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys
import os
sys.path.insert(0, '..')

master_pdir = "../plots/residuals_better/"
master_pdir = "../plots/tb_synth_residuals/"
os.mkdir(master_pdir)

def calc_best_fit(flat_samples):
    """
    Given a set of aligned (converted?) samples, calculate the median and
    errors of each parameter
    """
    return np.array( map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                     zip(*np.percentile(flat_samples, [16,50,84], axis=0))))

def getInfo(result, ix=-1):
    """
    "Result" is a numpy saved file from previous form of traceforward code
    """
    chain = result[1]
    flat_chain = chain.reshape(-1,9)
    flat_chain[:,6:8] = np.exp(flat_chain[:,6:8])
    best_fit = calc_best_fit(flat_chain)
    mean = best_fit[0]
    sigma = np.mean(best_fit[:,1:], axis=1)
    group_pars = result[4]
    return np.array([mean[ix], sigma[ix], group_pars[ix]])


# ------------------------------------------------------------
# ----------   Starting simple, lets plot age offsets  -------
# ------------------------------------------------------------

# I will use the result from the giant exhaustive run as a starting
# point, and alter it to accommodate new synth fits
"""
ages = [5, 10, 20]
spreads = [2, 5, 10]
v_disps = [1, 2, 5]
sizes = [25, 50, 100]
precs = ['perf', 'half', 'gaia', 'double']
ages = [5, 10]
spreads = [2]
v_disps = [1]
sizes   = [25]
precs = ['perf']
"""
ages = [5, 15, 30, 50]

o_spreads = [1, 5]
o_v_disps = [2, 10]
o_sizes = [25, 100]

e_spreads = [2, 10]
e_v_disps = [1, 5]
e_sizes = [50, 200]

precs = ['perf', 'half', 'gaia', 'double']
# load all the relevant data into a massive array where
# ix implicitly correspond to the value of the parameter

rdir = "../results/tb_synth_fit/"
chain_file = "final_chain.npy"
origin_file = "origins.npy"

prec_val = {'perf':0., 'half':0.5, 'gaia':1.0, 'double':2.0}

#------------------------------------------------------------
#-------   GATHER FITS FROM THE 'ODD' SIMULATIONS -----------
#------------------------------------------------------------
logging.info("Gathering 'ODD' simulations")
o_fits = np.zeros((len(ages), len(o_spreads), len(o_v_disps), len(o_sizes),
                    len(precs), 9, 3))
o_fits_w_errs = np.zeros((len(ages), len(o_spreads), len(o_v_disps), len(o_sizes),
                    len(precs), 9, 3))
for age_ix, age in enumerate(ages):
    for spread_ix, spread in enumerate(o_spreads):
        for v_disp_ix, v_disp in enumerate(o_v_disps):
            for size_ix, size in enumerate(o_sizes):
                for prec_ix, prec in enumerate(precs):
                    pdir = rdir + "{}_{}_{}_{}/{}/".format(
                        age, spread, v_disp, size, prec
                    )
                    try:
                        flat_chain = np.load(pdir + chain_file).\
                            reshape(-1,9)
                        conv_chain = np.copy(flat_chain)
                        conv_chain[:,6:8] = np.exp(conv_chain[:,6:8])
                        origin = np.load(pdir + origin_file).item()
                        fit_w_errs = calc_best_fit(conv_chain)
                        o_fits_w_errs[age_ix,spread_ix,v_disp_ix,size_ix,
                                      prec_ix] = fit_w_errs
                        means = fit_w_errs[:,0]
                        sigs = fit_w_errs[:,1:].mean(axis=1)
                        key_info = np.vstack((means,sigs,
                                              origin.pars[:-1])).T
                    except IOError:
                        print("Missing files for: {} {} {} {} {}".format(
                            age, spread, v_disp, size, prec
                        ))
                        key_info = np.array([None]*27).reshape(9,3)
                    o_fits[age_ix,spread_ix,v_disp_ix,size_ix,prec_ix]=\
                        key_info

o_res = (o_fits[:,:,:,:,:,:,0]-o_fits[:,:,:,:,:,:,2])
o_norm_res = o_res / o_fits[:,:,:,:,:,:,1]

# the fits where all parameters are within 15 sigma
o_worst_ixs = np.where(abs(o_norm_res).max(axis=-1) > 15)
o_bad_ixs = np.where(abs(o_norm_res).max(axis=-1) > 5)

o_fine_ixs = np.where(abs(o_norm_res).max(axis=-1) < 15)
o_great_ixs = np.where(abs(o_norm_res).max(axis=-1) < 5)

#problem_ix = np.array(np.where(abs(residuals) > 10)).T

# plot all great_ixs on 2D age, pos spread histogram
#plt.clf()
#plt.hist2d(o_norm_res[o_great_ixs][:,-1].flatten(),
#           o_norm_res[o_great_ixs][:,5].flatten(), bins=5)



#------------------------------------------------------------
#-------   GATHER FITS FROM THE 'EVEN' SIMULATIONS ----------
#------------------------------------------------------------
logging.info("Gathering 'EVEN' simulations")
e_fits = np.zeros((len(ages), len(e_spreads), len(e_v_disps), len(e_sizes),
                   len(precs), 9, 3))
e_fits_w_errs = np.zeros((len(ages), len(o_spreads), len(o_v_disps), len(o_sizes),
                          len(precs), 9, 3))
for age_ix, age in enumerate(ages):
    for spread_ix, spread in enumerate(e_spreads):
        for v_disp_ix, v_disp in enumerate(e_v_disps):
            for size_ix, size in enumerate(e_sizes):
                for prec_ix, prec in enumerate(precs):
                    pdir = rdir + "{}_{}_{}_{}/{}/".format(
                        age, spread, v_disp, size, prec
                    )
                    try:
                        flat_chain = np.load(pdir + chain_file). \
                            reshape(-1,9)
                        conv_chain = np.copy(flat_chain)
                        conv_chain[:,6:8] = np.exp(conv_chain[:,6:8])
                        origin = np.load(pdir + origin_file).item()
                        fit_w_errs = calc_best_fit(conv_chain)
                        e_fits_w_errs[age_ix,spread_ix,v_disp_ix,size_ix,
                                      prec_ix] = fit_w_errs
                        means = fit_w_errs[:,0]
                        sigs = fit_w_errs[:,1:].mean(axis=1)
                        key_info = np.vstack((means,sigs,
                                              origin.pars[:-1])).T
                    except IOError:
                        print("Missing files for: {} {} {} {} {}".format(
                            age, spread, v_disp, size, prec
                        ))
                        key_info = np.array([None]*27).reshape(9,3)
                    e_fits[age_ix,spread_ix,v_disp_ix,size_ix,prec_ix]= \
                        key_info

e_res = (e_fits[:,:,:,:,:,:,0]-e_fits[:,:,:,:,:,:,2])
e_norm_res = e_res / e_fits[:,:,:,:,:,:,1]

# the fits where all parameters are within 15 sigma
e_worst_ixs = np.where(abs(e_norm_res).max(axis=-1) > 15)
e_bad_ixs = np.where(abs(e_norm_res).max(axis=-1) > 5)

e_fine_ixs = np.where(abs(e_norm_res).max(axis=-1) < 15)
e_great_ixs = np.where(abs(e_norm_res).max(axis=-1) < 5)

# ------------------------------------------------------------
# ----  PLOTTING *ALL* NORMED RESIDUALS HIST    --------------
# ------------------------------------------------------------

plt.clf()
all_norm_resids = np.hstack((o_norm_res[:,:,:,:,:,-1].flatten(),
                             e_norm_res[:,:,:,:,:,-1].flatten()))
plt.hist(all_norm_resids)
plt.xlabel("Normalised offset in age")
plt.ylabel("Number of simulations")
plt.savefig(master_pdir + "all-norm-age-res-hist.pdf")

# ------------------------------------------------------------
# ----  PLOTTING RAW RESIDUALS WITH LOW DV HIST  -------------
# ------------------------------------------------------------

plt.clf()
all_low_dv_raw_resids = np.hstack((o_res[:,:,0,:,:,-1].flatten(),
                                   e_res[:,:,0,:,:,-1].flatten()))
plt.hist(all_low_dv_raw_resids)
plt.xlabel("Raw age offset [Myr]")
plt.ylabel("Number of simulations")
plt.savefig(master_pdir + "all-raw-age-res-low-dv-hist.pdf")

# ------------------------------------------------------------
# ----  PLOTTING *ALL* NORMED RESIDUALS V PARS  --------------
# ------------------------------------------------------------

plt.clf()
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(7,7), sharey=True)

# PLOTTING ALL RESIDUALS AS FUNCTION OF TRUE AGE
ax = axs[0,0]
age_fit_means = []
age_fit_stds = []

for i, age in enumerate(ages):
    age_norm_resids = np.hstack((o_norm_res[i,:,:,:,:,-1].flatten(),
                             e_norm_res[i,:,:,:,:,-1].flatten()))
    age_fit_means.append(np.mean(age_norm_resids))
    age_fit_stds.append(np.std(age_norm_resids))


ax.errorbar(ages, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(ages, np.zeros(len(ages)), color='b', ls=':')
ax.set_xlabel("True age [Myr]")
ax.set_ylabel("Normalised offset in age")

# PLOTTING ALL RESIDUALS AS FUNCTION OF STAR COUNT
ax = axs[0,1]
age_fit_means = []
age_fit_stds = []
sizes = []

for i, size in enumerate(zip(o_sizes, e_sizes)):
    age_norm_resids = o_norm_res[:,:,:,:,i,-1]
    age_fit_means.append(np.mean(age_norm_resids))
    age_fit_stds.append(np.std(age_norm_resids))
    sizes.append(size[0])

    age_norm_resids = e_norm_res[:,:,:,:,i,-1]
    age_fit_means.append(np.mean(age_norm_resids))
    age_fit_stds.append(np.std(age_norm_resids))
    sizes.append(size[1])


ax.errorbar(sizes, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(sizes, np.zeros(len(sizes)), color='b', ls=':')
ax.set_xlabel("Star count")

# PLOTTING ALL RESIDUALS AS FUNCTION OF DX
ax = axs[1,0]
age_fit_means = []
age_fit_stds = []
spreads = []

for i, spread in enumerate(zip(o_spreads, e_spreads)):
    age_norm_resids = o_norm_res[:,i,:,:,:,-1]
    age_fit_means.append(np.mean(age_norm_resids))
    age_fit_stds.append(np.std(age_norm_resids))
    spreads.append(spread[0])

    age_norm_resids = e_norm_res[:,i,:,:,:,-1]
    age_fit_means.append(np.mean(age_norm_resids))
    age_fit_stds.append(np.std(age_norm_resids))
    spreads.append(spread[1])

ax.errorbar(spreads, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(spreads, np.zeros(len(spreads)), color='b', ls=':')
ax.set_xlabel("Spread [pc]")
ax.set_ylabel("Normalised offset in age")

# PLOTTING ALL RESIDUALS AS FUNCTION OF DV
ax = axs[1,1]
age_fit_means = []
age_fit_stds = []
v_disps = []

for i, v_disp in enumerate(zip(o_v_disps, e_v_disps)):
    # Flipped the order cause e_v_disps has the lower v_disps
    age_norm_resids = e_norm_res[:,:,i,:,:,-1]
    age_fit_means.append(np.mean(age_norm_resids))
    age_fit_stds.append(np.std(age_norm_resids))
    v_disps.append(e_v_disps[i])

    age_norm_resids = o_norm_res[:,:,i,:,:,-1]
    age_fit_means.append(np.mean(age_norm_resids))
    age_fit_stds.append(np.std(age_norm_resids))
    v_disps.append(o_v_disps[i])

ax.errorbar(v_disps, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(v_disps, np.zeros(len(v_disps)), color='b', ls=':')
ax.set_xlabel("Velocity dispersion [km/s]")


# PLOTTING ALL RESIDUALS AS FUNCTION OF PRECISION
ax = axs[2,0]
age_fit_means = []
age_fit_stds = []
vals = []

for i, prec in enumerate(precs):
    vals.append(prec_val[prec])
    age_norm_resids = np.hstack((o_norm_res[:,:,:,:,i,-1].flatten(),
                            e_norm_res[:,:,:,:,i,-1].flatten()))
    age_fit_means.append(np.mean(age_norm_resids))
    age_fit_stds.append(np.std(age_norm_resids))

ax.errorbar(vals, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(vals, np.zeros(len(vals)), color='b', ls=':')
ax.set_xlabel("Fraction of Gaia error")
ax.set_ylabel("Normalised offset in age")

axs[2,1].remove()

fig.tight_layout()
plt.savefig(master_pdir + "normed-age-res-all.pdf")

# ------------------------------------------------------------
# ----  PLOTTING LOW DV NORMED RESIDUALS V PARS  -------------
# ------------------------------------------------------------

plt.clf()
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(7,7), sharey=True)

# PLOTTING ALL RESIDUALS AS FUNCTION OF TRUE AGE
ax = axs[0,0]
age_fit_means = []
age_fit_stds = []

for i, age in enumerate(ages):
    age_norm_resids = np.hstack((o_norm_res[i,:,0,:,:,-1].flatten(),
                             e_norm_res[i,:,0,:,:,-1].flatten()))
    age_fit_means.append(np.mean(age_norm_resids))
    age_fit_stds.append(np.std(age_norm_resids))


ax.errorbar(ages, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(ages, np.zeros(len(ages)), color='b', ls=':')
ax.set_xlabel("True age [Myr]")
ax.set_ylabel("Normalised offset in age")

# PLOTTING ALL RESIDUALS AS FUNCTION OF STAR COUNT
ax = axs[0,1]
age_fit_means = []
age_fit_stds = []
sizes = []

for i, size in enumerate(zip(o_sizes, e_sizes)):
    age_norm_resids = o_norm_res[:,:,0,:,i,-1]
    age_fit_means.append(np.mean(age_norm_resids))
    age_fit_stds.append(np.std(age_norm_resids))
    sizes.append(size[0])

    age_norm_resids = e_norm_res[:,:,0,:,i,-1]
    age_fit_means.append(np.mean(age_norm_resids))
    age_fit_stds.append(np.std(age_norm_resids))
    sizes.append(size[1])


ax.errorbar(sizes, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(sizes, np.zeros(len(sizes)), color='b', ls=':')
ax.set_xlabel("Star count")



# PLOTTING ALL RESIDUALS AS FUNCTION OF DX
ax = axs[1,0]
age_fit_means = []
age_fit_stds = []
spreads = []

for i, spread in enumerate(zip(o_spreads, e_spreads)):
    age_norm_resids = o_norm_res[:,i,0,:,:,-1]
    age_fit_means.append(np.mean(age_norm_resids))
    age_fit_stds.append(np.std(age_norm_resids))
    spreads.append(spread[0])

    age_norm_resids = e_norm_res[:,i,0,:,:,-1]
    age_fit_means.append(np.mean(age_norm_resids))
    age_fit_stds.append(np.std(age_norm_resids))
    spreads.append(spread[1])


ax.errorbar(spreads, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(spreads, np.zeros(len(spreads)), color='b', ls=':')
ax.set_xlabel("Spread [pc]")
ax.set_ylabel("Normalised offset in age")

# PLOTTING ALL RESIDUALS AS FUNCTION OF DV
ax = axs[1,1]
age_fit_means = []
age_fit_stds = []
v_disps = []

#for i, v_disp in enumerate(zip(o_v_disps, e_v_disps)):
age_norm_resids = o_norm_res[:,:,0,:,:,-1]
age_fit_means.append(np.mean(age_norm_resids))
age_fit_stds.append(np.std(age_norm_resids))
v_disps.append(o_v_disps[0])

age_norm_resids = e_norm_res[:,:,0,:,:,-1]
age_fit_means.append(np.mean(age_norm_resids))
age_fit_stds.append(np.std(age_norm_resids))
v_disps.append(e_v_disps[0])


ax.errorbar(v_disps, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(v_disps, np.zeros(len(v_disps)), color='b', ls=':')
ax.set_xlabel("Velocity dispersion [km/s]")


# PLOTTING ALL RESIDUALS AS FUNCTION OF PRECISION
ax = axs[2,0]
age_fit_means = []
age_fit_stds = []
vals = []

for i, prec in enumerate(precs):
    vals.append(prec_val[prec])
    age_norm_resids = np.hstack((o_norm_res[:,:,0,:,i,-1].flatten(),
                            e_norm_res[:,:,0,:,i,-1].flatten()))
    age_fit_means.append(np.mean(age_norm_resids))
    age_fit_stds.append(np.std(age_norm_resids))

ax.errorbar(vals, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(vals, np.zeros(len(vals)), color='b', ls=':')
ax.set_xlabel("Fraction of Gaia error")
ax.set_ylabel("Normalised offset in age")

axs[2,1].remove()

fig.tight_layout()
plt.savefig(master_pdir + "normed-age-res-low-dv.pdf")

# ------------------------------------------------------------
# -------  PLOTTING LOW DV RAW RESIDUALS V PARS  -------------
# ------------------------------------------------------------



plt.clf()
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(7,7), sharey=True)

# PLOTTING LOW DV RAW RESIDUALS AS FUNCTION OF TRUE AGE
ax = axs[0,0]
age_fit_means = []
age_fit_stds = []

for i, age in enumerate(ages):
    age_resids = np.hstack((o_res[i,:,0,:,:,-1].flatten(),
                             e_res[i,:,0,:,:,-1].flatten()))
    age_fit_means.append(np.mean(age_resids))
    age_fit_stds.append(np.std(age_resids))


ax.errorbar(ages, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(ages, np.zeros(len(ages)), color='b', ls=':')
ax.set_xlabel("True age [Myr]")
ax.set_ylabel("Raw offset in age [Myr]")

# PLOTTING LOW DV RAW RESIDUALS AS FUNCTION OF STAR COUNT
ax = axs[0,1]
age_fit_means = []
age_fit_stds = []
sizes = []

for i, size in enumerate(zip(o_sizes, e_sizes)):
    age_resids = o_res[:,:,0,:,i,-1]
    age_fit_means.append(np.mean(age_resids))
    age_fit_stds.append(np.std(age_resids))
    sizes.append(size[0])

    age_resids = e_res[:,:,0,:,i,-1]
    age_fit_means.append(np.mean(age_resids))
    age_fit_stds.append(np.std(age_resids))
    sizes.append(size[1])


ax.errorbar(sizes, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(sizes, np.zeros(len(sizes)), color='b', ls=':')
ax.set_xlabel("Star count")

# PLOTTING LOW DV RAW RESIDUALS AS FUNCTION OF DX
ax = axs[1,0]
age_fit_means = []
age_fit_stds = []
spreads = []

for i, spread in enumerate(zip(o_spreads, e_spreads)):
    age_resids = o_res[:,i,0,:,:,-1]
    age_fit_means.append(np.mean(age_resids))
    age_fit_stds.append(np.std(age_resids))
    spreads.append(spread[0])

    age_resids = e_res[:,i,0,:,:,-1]
    age_fit_means.append(np.mean(age_resids))
    age_fit_stds.append(np.std(age_resids))
    spreads.append(spread[1])

ax.errorbar(spreads, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(spreads, np.zeros(len(spreads)), color='b', ls=':')
ax.set_xlabel("Spread [pc]")
ax.set_ylabel("Raw offset in age [Myr]")

# PLOTTING LOW DV RAW RESIDUALS AS FUNCTION OF DV
ax = axs[1,1]
age_fit_means = []
age_fit_stds = []
v_disps = []

#for i, v_disp in enumerate(zip(o_v_disps, e_v_disps)):
age_resids = o_res[:,:,0,:,:,-1]
age_fit_means.append(np.mean(age_resids))
age_fit_stds.append(np.std(age_resids))
v_disps.append(o_v_disps[0])

age_resids = e_res[:,:,0,:,:,-1]
age_fit_means.append(np.mean(age_resids))
age_fit_stds.append(np.std(age_resids))
v_disps.append(e_v_disps[0])


ax.errorbar(v_disps, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(v_disps, np.zeros(len(v_disps)), color='b', ls=':')
ax.set_xlabel("Velocity dispersion [km/s]")


# PLOTTING LOW DV RAW RESIDUALS AS FUNCTION OF PRECISION
ax = axs[2,0]
age_fit_means = []
age_fit_stds = []
vals = []

for i, prec in enumerate(precs):
    vals.append(prec_val[prec])
    age_resids = np.hstack((o_res[:,:,0,:,i,-1].flatten(),
                            e_res[:,:,0,:,i,-1].flatten()))
    age_fit_means.append(np.mean(age_resids))
    age_fit_stds.append(np.std(age_resids))

ax.errorbar(vals, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(vals, np.zeros(len(vals)), color='b', ls=':')
ax.set_xlabel("Fraction of Gaia error")
ax.set_ylabel("Raw offset in age [Myr]")

axs[2,1].remove()

fig.tight_layout()
plt.savefig(master_pdir + "raw-age-res-low-dv.pdf")


# ------------------------------------------------------------
# ----  PLOTTING LOW DV LOW DX RAW RESIDUALS V PARS  ---------
# ------------------------------------------------------------

plt.clf()
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(7,7), sharey=True)

# PLOTTING LOW DV RAW RESIDUALS AS FUNCTION OF TRUE AGE
ax = axs[0,0]
age_fit_means = []
age_fit_stds = []

for i, age in enumerate(ages):
    age_resids = np.hstack((o_res[i,0,0,:,:,-1].flatten(),
                             e_res[i,0,0,:,:,-1].flatten()))
    age_fit_means.append(np.mean(age_resids))
    age_fit_stds.append(np.std(age_resids))


ax.errorbar(ages, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(ages, np.zeros(len(ages)), color='b', ls=':')
ax.set_xlabel("True age [Myr]")
ax.set_ylabel("Raw offset in age [Myr]")

# PLOTTING LOW DV RAW RESIDUALS AS FUNCTION OF STAR COUNT
ax = axs[0,1]
age_fit_means = []
age_fit_stds = []
sizes = []

for i, size in enumerate(zip(o_sizes, e_sizes)):
    age_resids = o_res[:,0,0,:,i,-1]
    age_fit_means.append(np.mean(age_resids))
    age_fit_stds.append(np.std(age_resids))
    sizes.append(size[0])

    age_resids = e_res[:,0,0,:,i,-1]
    age_fit_means.append(np.mean(age_resids))
    age_fit_stds.append(np.std(age_resids))
    sizes.append(size[1])


ax.errorbar(sizes, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(sizes, np.zeros(len(sizes)), color='b', ls=':')
ax.set_xlabel("Star count")

# PLOTTING LOW DV RAW RESIDUALS AS FUNCTION OF DX
ax = axs[1,0]
age_fit_means = []
age_fit_stds = []
spreads = []

#for i, spread in enumerate(zip(o_spreads, e_spreads)):
age_resids = o_res[:,0,0,:,:,-1]
age_fit_means.append(np.mean(age_resids))
age_fit_stds.append(np.std(age_resids))
spreads.append(o_spreads[0])

age_resids = e_res[:,0,0,:,:,-1]
age_fit_means.append(np.mean(age_resids))
age_fit_stds.append(np.std(age_resids))
spreads.append(e_spreads[0])

#age_resids = o_res[:,1,0,:,:,-1]
#age_fit_means.append(np.mean(age_resids))
#age_fit_stds.append(np.std(age_resids))
#spreads.append(o_spreads[1])

ax.errorbar(spreads, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(spreads, np.zeros(len(spreads)), color='b', ls=':')
ax.set_xlabel("Spread [pc]")
ax.set_ylabel("Raw offset in age [Myr]")

# PLOTTING LOW DV RAW RESIDUALS AS FUNCTION OF DV
ax = axs[1,1]
age_fit_means = []
age_fit_stds = []
v_disps = []

#for i, v_disp in enumerate(zip(o_v_disps, e_v_disps)):
age_resids = o_res[:,0,0,:,:,-1]
age_fit_means.append(np.mean(age_resids))
age_fit_stds.append(np.std(age_resids))
v_disps.append(o_v_disps[0])

age_resids = e_res[:,0,0,:,:,-1]
age_fit_means.append(np.mean(age_resids))
age_fit_stds.append(np.std(age_resids))
v_disps.append(e_v_disps[0])


ax.errorbar(v_disps, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(v_disps, np.zeros(len(v_disps)), color='b', ls=':')
ax.set_xlabel("Velocity dispersion [km/s]")


# PLOTTING LOW DV RAW RESIDUALS AS FUNCTION OF PRECISION
ax = axs[2,0]
age_fit_means = []
age_fit_stds = []
vals = []

for i, prec in enumerate(precs):
    vals.append(prec_val[prec])
    age_resids = np.hstack((o_res[:,0,0,:,i,-1].flatten(),
                            e_res[:,0,0,:,i,-1].flatten()))
    age_fit_means.append(np.mean(age_resids))
    age_fit_stds.append(np.std(age_resids))

ax.errorbar(vals, age_fit_means, yerr=age_fit_stds, fmt='b.')
ax.plot(vals, np.zeros(len(vals)), color='b', ls=':')
ax.set_xlabel("Fraction of Gaia error")
ax.set_ylabel("Raw offset in age [Myr]")

axs[2,1].remove()

fig.tight_layout()
plt.savefig(master_pdir + "raw-age-res-low-dv-low-dx.pdf")

# # ------------------------------------------------------------
# # ----  PLOTTING LOW-DV NORMED RESIDUALS V PARS  -------------
# # ------------------------------------------------------------
#
# # PLOTTING ALL RESIDUALS AS FUNCTION OF TRUE AGE
# age_fit_means = []
# age_fit_stds = []
#
# for i, age in enumerate(ages):
#     age_norm_resids = np.hstack((o_norm_res[i,:,0,:,:,-1].flatten(),
#                              e_norm_res[i,:,0,:,:,-1].flatten()))
#     age_fit_means.append(np.mean(age_norm_resids))
#     age_fit_stds.append(np.std(age_norm_resids))
# plt.clf()
# plt.errorbar(ages, age_fit_means, yerr=age_fit_stds, fmt='b.')
# plt.plot(ages, np.zeros(len(ages)), color='b', ls=':')
# plt.xlabel("True age [Myr]")
# plt.ylabel("Normalised offset in age")
# plt.savefig(master_pdir + "age-norm-residuals-age-low-dv.pdf")
#
# # PLOTTING ALL RESIDUALS AS FUNCTION OF STAR COUNT
# age_fit_means = []
# age_fit_stds = []
# sizes = []
#
# for i, size in enumerate(zip(o_sizes, e_sizes)):
#     age_norm_resids = o_norm_res[:,:,0,:,i,-1]
#     age_fit_means.append(np.mean(age_norm_resids))
#     age_fit_stds.append(np.std(age_norm_resids))
#     sizes.append(size[0])
#
#     age_norm_resids = e_norm_res[:,:,0,:,i,-1]
#     age_fit_means.append(np.mean(age_norm_resids))
#     age_fit_stds.append(np.std(age_norm_resids))
#     sizes.append(size[1])
#
# plt.clf()
# plt.errorbar(sizes, age_fit_means, yerr=age_fit_stds, fmt='b.')
# plt.plot(sizes, np.zeros(len(sizes)), color='b', ls=':')
# plt.xlabel("Star count")
# plt.ylabel("Normalised offset in age")
# plt.savefig(master_pdir + "age-norm-residuals-size-low-dv.pdf")
#
# # PLOTTING ALL RESIDUALS AS FUNCTION OF PRECISION
# age_fit_means = []
# age_fit_stds = []
# vals = []
#
# for i, prec in enumerate(precs):
#     vals.append(prec_val[prec])
#     age_norm_resids = np.hstack((o_norm_res[:,:,0,:,i,-1].flatten(),
#                             e_norm_res[:,:,0,:,i,-1].flatten()))
#     age_fit_means.append(np.mean(age_norm_resids))
#     age_fit_stds.append(np.std(age_norm_resids))
# plt.clf()
# plt.errorbar(vals, age_fit_means, yerr=age_fit_stds, fmt='b.')
# plt.plot(vals, np.zeros(len(vals)), color='b', ls=':')
# plt.xlabel("Fraction of Gaia error")
# plt.ylabel("Normalised offset in age")
# plt.savefig(master_pdir + "age-norm-residuals-prec-low-dv.pdf")
#
#
# # PLOTTING ALL RESIDUALS AS FUNCTION OF DX
# age_fit_means = []
# age_fit_stds = []
# spreads = []
#
# for i, spread in enumerate(zip(o_spreads, e_spreads)):
#     age_norm_resids = o_norm_res[:,i,0,:,:,-1]
#     age_fit_means.append(np.mean(age_norm_resids))
#     age_fit_stds.append(np.std(age_norm_resids))
#     spreads.append(spread[0])
#
#     age_norm_resids = e_norm_res[:,i,0,:,:,-1]
#     age_fit_means.append(np.mean(age_norm_resids))
#     age_fit_stds.append(np.std(age_norm_resids))
#     spreads.append(spread[1])
#
# plt.clf()
# plt.errorbar(spreads, age_fit_means, yerr=age_fit_stds, fmt='b.')
# plt.plot(spreads, np.zeros(len(spreads)), color='b', ls=':')
# plt.xlabel("Spread [pc]")
# plt.ylabel("Normalised offset in age")
# plt.savefig(master_pdir + "age-norm-residuals-spread-low-dv.pdf")
#
# # PLOTTING ALL RESIDUALS AS FUNCTION OF DV
# age_fit_means = []
# age_fit_stds = []
# v_disps = []

# # for i, v_disp in enumerate(zip(o_v_disps, e_v_disps)):
# #    age_norm_resids = o_norm_res[:,:,i,:,:,-1]
# #    age_fit_means.append(np.mean(age_norm_resids))
# #    age_fit_stds.append(np.std(age_norm_resids))
# #    v_disps.append(o_v_disps[i])
# #
# #    age_norm_resids = e_norm_res[:,:,i,:,:,-1]
# #    age_fit_means.append(np.mean(age_norm_resids))
# #    age_fit_stds.append(np.std(age_norm_resids))
# #    v_disps.append(e_v_disps[i])
# #
# # plt.clf()
# # plt.errorbar(v_disps, age_fit_means, yerr=age_fit_stds, fmt='b.')
# # plt.plot(v_disps, np.zeros(len(v_disps)), color='b', ls=':')
# # plt.xlabel("Velocity dispersion [km/s]")
# # plt.ylabel("Normalised offset in age")
# # plt.savefig(master_pdir + "age-norm-residuals-v-disp.pdf")

# # ------------------------------------------------------------
# # --------   PLOTTING RAW RESIDUALS V PARS  ------------------
# # ------------------------------------------------------------
#
# # PLOTTING RAW RESIDUALS AS FUNCTION OF TRUE AGE
# age_fit_means = []
# age_fit_stds = []
#
# for i, age in enumerate(ages):
#     age_resids = np.hstack((o_res[i,:,0,:,:,-1].flatten(),
#                              e_res[i,:,0,:,:,-1].flatten()))
#     age_fit_means.append(np.mean(age_resids))
#     age_fit_stds.append(np.std(age_resids))
# plt.clf()
# plt.errorbar(ages, age_fit_means, yerr=age_fit_stds, fmt='b.')
# plt.plot(ages, np.zeros(len(ages)), color='b', ls=':')
# plt.xlabel("True age [Myr]")
# plt.ylabel("Offset in age [Myr]")
# plt.savefig(master_pdir + "age-residuals-age.pdf")
#
# # PLOTTING RAW RESIDUALS AS FUNCTION OF STAR COUNT
# age_fit_means = []
# age_fit_stds = []
# sizes = []
#
# for i, size in enumerate(zip(o_sizes, e_sizes)):
#     age_resids = o_res[:,:,0,:,i,-1]
#     age_fit_means.append(np.mean(age_resids))
#     age_fit_stds.append(np.std(age_resids))
#     sizes.append(size[0])
#
#     age_resids = e_res[:,:,0,:,i,-1]
#     age_fit_means.append(np.mean(age_resids))
#     age_fit_stds.append(np.std(age_resids))
#     sizes.append(size[1])
#
# plt.clf()
# plt.errorbar(sizes, age_fit_means, yerr=age_fit_stds, fmt='b.')
# plt.plot(sizes, np.zeros(len(sizes)), color='b', ls=':')
# plt.xlabel("Star count")
# plt.ylabel("Offset in age [Myr]")
# plt.savefig(master_pdir + "age-residuals-size.pdf")
#
# # PLOTTING RAW RESIDUALS AS FUNCTION OF PRECISION
# age_fit_means = []
# age_fit_stds = []
# vals = []
#
# for i, prec in enumerate(precs):
#     vals.append(prec_val[prec])
#     age_resids = np.hstack((o_res[:,:,0,:,i,-1].flatten(),
#                             e_res[:,:,0,:,i,-1].flatten()))
#     age_fit_means.append(np.mean(age_resids))
#     age_fit_stds.append(np.std(age_resids))
# plt.clf()
# plt.errorbar(vals, age_fit_means, yerr=age_fit_stds, fmt='b.')
# plt.plot(vals, np.zeros(len(vals)), color='b', ls=':')
# plt.xlabel("Fraction of Gaia error")
# plt.ylabel("Offset in age [Myr]")
# plt.savefig(master_pdir + "age-residuals-prec.pdf")
#
#
# # PLOTTING RAW RESIDUALS AS FUNCTION OF DX
# age_fit_means = []
# age_fit_stds = []
# spreads = []
#
# for i, spread in enumerate(zip(o_spreads, e_spreads)):
#     age_resids = o_res[:,i,0,:,:,-1]
#     age_fit_means.append(np.mean(age_resids))
#     age_fit_stds.append(np.std(age_resids))
#     spreads.append(spread[0])
#
#     age_resids = e_res[:,i,0,:,:,-1]
#     age_fit_means.append(np.mean(age_resids))
#     age_fit_stds.append(np.std(age_resids))
#     spreads.append(spread[1])
#
# plt.clf()
# plt.errorbar(spreads, age_fit_means, yerr=age_fit_stds, fmt='b.')
# plt.plot(spreads, np.zeros(len(spreads)), color='b', ls=':')
# plt.xlabel("Spread [pc]")
# plt.ylabel("Offset in age [Myr]")
# plt.savefig(master_pdir + "age-residuals-spread.pdf")
#
# # PLOTTING RAW RESIDUALS AS FUNCTION OF DV
# age_fit_means = []
# age_fit_stds = []
# v_disps = []
#
# #for i, v_disp in enumerate(zip(o_v_disps, e_v_disps)):
# age_resids = o_res[:,:,0,:,:,-1]
# age_fit_means.append(np.mean(age_resids))
# age_fit_stds.append(np.std(age_resids))
# v_disps.append(o_v_disps[0])
#
# age_resids = e_res[:,:,0,:,:,-1]
# age_fit_means.append(np.mean(age_resids))
# age_fit_stds.append(np.std(age_resids))
# v_disps.append(e_v_disps[0])
#
# plt.clf()
# plt.errorbar(v_disps, age_fit_means, yerr=age_fit_stds, fmt='b.')
# plt.plot(v_disps, np.zeros(len(v_disps)), color='b', ls=':')
# plt.xlabel("Velocity dispersion [km/s]")
# plt.ylabel("Offset in age [Myr]")
# plt.savefig(master_pdir + "age-residuals-v-disp.pdf")
#
# # ------------------------------------------------------------
# # --------   REMOVING DX=10 FROM ALL FITS  -------------------
# # ------------------------------------------------------------
#
# # PLOTTING RAW RESIDUALS AS FUNCTION OF TRUE AGE
# age_fit_means = []
# age_fit_stds = []
#
# for i, age in enumerate(ages):
#     age_resids = np.hstack((o_res[i,:,0,:,:,-1].flatten(),
#                              e_res[i,0,0,:,:,-1].flatten()))
#     age_fit_means.append(np.mean(age_resids))
#     age_fit_stds.append(np.std(age_resids))
# plt.clf()
# plt.errorbar(ages, age_fit_means, yerr=age_fit_stds, fmt='b.')
# plt.plot(ages, np.zeros(len(ages)), color='b', ls=':')
# plt.xlabel("True age [Myr]")
# plt.ylabel("Offset in age [Myr]")
# plt.savefig(master_pdir + "age-residuals-age-low-dx.pdf")
#
# # PLOTTING RAW RESIDUALS AS FUNCTION OF STAR COUNT
# age_fit_means = []
# age_fit_stds = []
# sizes = []
#
# for i, size in enumerate(zip(o_sizes, e_sizes)):
#     age_resids = o_res[:,:,0,:,i,-1]
#     age_fit_means.append(np.mean(age_resids))
#     age_fit_stds.append(np.std(age_resids))
#     sizes.append(size[0])
#
#     age_resids = e_res[:,0,0,:,i,-1]
#     age_fit_means.append(np.mean(age_resids))
#     age_fit_stds.append(np.std(age_resids))
#     sizes.append(size[1])
#
# plt.clf()
# plt.errorbar(sizes, age_fit_means, yerr=age_fit_stds, fmt='b.')
# plt.plot(sizes, np.zeros(len(sizes)), color='b', ls=':')
# plt.xlabel("Star count")
# plt.ylabel("Offset in age [Myr]")
# plt.savefig(master_pdir + "age-residuals-size-low-dx.pdf")
#
# # PLOTTING RAW RESIDUALS AS FUNCTION OF PRECISION
# age_fit_means = []
# age_fit_stds = []
# vals = []
#
# for i, prec in enumerate(precs):
#     vals.append(prec_val[prec])
#     age_resids = np.hstack((o_res[:,:,0,:,i,-1].flatten(),
#                             e_res[:,0,0,:,i,-1].flatten()))
#     age_fit_means.append(np.mean(age_resids))
#     age_fit_stds.append(np.std(age_resids))
# plt.clf()
# plt.errorbar(vals, age_fit_means, yerr=age_fit_stds, fmt='b.')
# plt.plot(vals, np.zeros(len(vals)), color='b', ls=':')
# plt.xlabel("Fraction of Gaia error")
# plt.ylabel("Offset in age [Myr]")
# plt.savefig(master_pdir + "age-residuals-prec-low-dx.pdf")
#
#
# # PLOTTING RAW RESIDUALS AS FUNCTION OF DV
# age_fit_means = []
# age_fit_stds = []
# v_disps = []
#
# #for i, v_disp in enumerate(zip(o_v_disps, e_v_disps)):
# age_resids = o_res[:,:,0,:,:,-1]
# age_fit_means.append(np.mean(age_resids))
# age_fit_stds.append(np.std(age_resids))
# v_disps.append(o_v_disps[0])
#
# age_resids = e_res[:,0,0,:,:,-1]
# age_fit_means.append(np.mean(age_resids))
# age_fit_stds.append(np.std(age_resids))
# v_disps.append(e_v_disps[0])
#
# plt.clf()
# plt.errorbar(v_disps, age_fit_means, yerr=age_fit_stds, fmt='b.')
# plt.plot(v_disps, np.zeros(len(v_disps)), color='b', ls=':')
# plt.xlabel("Velocity dispersion [km/s]")
# plt.ylabel("Offset in age [Myr]")
# plt.savefig(master_pdir + "age-residuals-v-disp-low-dx.pdf")
#

# # FOR ALL LOW DV PLOT 2D HISTOGRAMS OF DX AND DV, CUT BY EACH PARAMETER
# plt.clf()
# plt.hist2d(np.hstack((e_norm_res[:2,:,0,:,:,6].flatten(),
#                       o_norm_res[:2,:,0,:,:,6].flatten())),
#            np.hstack((e_norm_res[:2,:,0,:,:,7].flatten(),
#                       o_norm_res[:2,:,0,:,:,7].flatten())),
#            bins=5)
# plt.xlabel("Normalised Residual in dX")
# plt.ylabel("Normalised Residual in dV")
# plt.savefig(master_pdir + "dx-dv-age-low.pdf")
#
# plt.clf()
# plt.hist2d(np.hstack((e_norm_res[2:,:,0,:,:,6].flatten(),
#                       o_norm_res[2:,:,0,:,:,6].flatten())),
#            np.hstack((e_norm_res[2:,:,0,:,:,7].flatten(),
#                       o_norm_res[2:,:,0,:,:,7].flatten())),
#            bins=5)
# plt.xlabel("Normalised Residual in dX")
# plt.ylabel("Normalised Residual in dV")
# plt.savefig(master_pdir + "dx-dv-age-high.pdf")
#
# plt.clf()
# plt.hist2d(np.hstack((e_norm_res[:,0,0,:,:,6].flatten(),
#                       o_norm_res[:,0,0,:,:,6].flatten())),
#            np.hstack((e_norm_res[:,0,0,:,:,7].flatten(),
#                       o_norm_res[:,0,0,:,:,7].flatten())),
#            bins=5)
# plt.xlabel("Normalised Residual in dX")
# plt.ylabel("Normalised Residual in dV")
# plt.savefig(master_pdir + "dx-dv-dx-low.pdf")
#
# plt.clf()
# plt.hist2d(np.hstack((e_norm_res[:,1,0,:,:,6].flatten(),
#                       o_norm_res[:,1,0,:,:,6].flatten())),
#            np.hstack((e_norm_res[:,1,0,:,:,7].flatten(),
#                       o_norm_res[:,1,0,:,:,7].flatten())),
#            bins=5)
# plt.xlabel("Normalised Residual in dX")
# plt.ylabel("Normalised Residual in dV")
# plt.savefig(master_pdir + "dx-dv-dx-high.pdf")
#
# plt.clf()
# plt.hist2d(o_norm_res[:,:,0,:,:,6].flatten(),
#            o_norm_res[:,:,0,:,:,7].flatten(),
#            bins=5)
# plt.xlabel("Normalised Residual in dX")
# plt.ylabel("Normalised Residual in dV")
# plt.savefig(master_pdir + "dx-dv-dv-low.pdf")
#
# plt.clf()
# plt.hist2d(e_norm_res[:,:,0,:,:,6].flatten(),
#            e_norm_res[:,:,0,:,:,7].flatten(),
#            bins=5)
# plt.xlabel("Normalised Residual in dX")
# plt.ylabel("Normalised Residual in dV")
# plt.savefig(master_pdir + "dx-dv-dv-high.pdf")
#
#
# plt.clf()
# plt.hist2d(np.hstack((e_norm_res[:,:,0,0,:,6].flatten(),
#                       o_norm_res[:,:,0,0,:,6].flatten())),
#            np.hstack((e_norm_res[:,:,0,0,:,7].flatten(),
#                       o_norm_res[:,:,0,0,:,7].flatten())),
#            bins=5)
# plt.xlabel("Normalised Residual in dX")
# plt.ylabel("Normalised Residual in dV")
# plt.savefig(master_pdir + "dx-dv-count-low.pdf")
#
# plt.clf()
# plt.hist2d(np.hstack((e_norm_res[:,:,0,1,:,6].flatten(),
#                       o_norm_res[:,:,0,1,:,6].flatten())),
#            np.hstack((e_norm_res[:,:,0,1,:,7].flatten(),
#                       o_norm_res[:,:,0,1,:,7].flatten())),
#            bins=5)
# plt.xlabel("Normalised Residual in dX")
# plt.ylabel("Normalised Residual in dV")
# plt.savefig(master_pdir + "dx-dv-count-high.pdf")
#
# plt.clf()
# plt.hist2d(np.hstack((e_norm_res[:,:,0,:,0,6].flatten(),
#                       o_norm_res[:,:,0,:,0,6].flatten())),
#            np.hstack((e_norm_res[:,:,0,:,0,7].flatten(),
#                       o_norm_res[:,:,0,:,0,7].flatten())),
#            bins=5)
# plt.xlabel("Normalised Residual in dX")
# plt.ylabel("Normalised Residual in dV")
# plt.savefig(master_pdir + "dx-dv-prec-perf.pdf")
#
# plt.clf()
# plt.hist2d(np.hstack((e_norm_res[:,:,0,:,1,6].flatten(),
#                       o_norm_res[:,:,0,:,1,6].flatten())),
#            np.hstack((e_norm_res[:,:,0,:,1,7].flatten(),
#                       o_norm_res[:,:,0,:,1,7].flatten())),
#            bins=5)
# plt.xlabel("Normalised Residual in dX")
# plt.ylabel("Normalised Residual in dV")
# plt.savefig(master_pdir + "dx-dv-prec-half.pdf")
#
# plt.clf()
# plt.hist2d(np.hstack((e_norm_res[:,:,0,:,2,6].flatten(),
#                       o_norm_res[:,:,0,:,2,6].flatten())),
#            np.hstack((e_norm_res[:,:,0,:,2,7].flatten(),
#                       o_norm_res[:,:,0,:,2,7].flatten())),
#            bins=5)
# plt.xlabel("Normalised Residual in dX")
# plt.ylabel("Normalised Residual in dV")
# plt.savefig(master_pdir + "dx-dv-prec-gaia.pdf")
#
# # MAIN CONTRIBUTORS TO DX OFFSET:
# # - count (higher is better)
# # - dv (higher is better (that is, 1km/s vs 2km/s))
# # - precision (perf is better)
#
# plt.clf()
# plt.hist(e_norm_res[:,:,0,1,:2,6].flatten(), bins=4)
# plt.xlabel("Normalised Residual in dX")
# plt.savefig(master_pdir + "best-dx.pdf")
#
# # MAIN CONTRIBUTORS TO DV OFFSET:
# # - count (higher is better)
# # - dx (smaller is better)
#
# plt.clf()
# plt.hist(np.hstack((o_norm_res[:,0,0,1,:,7].flatten(),
#                     e_norm_res[:,0,0,1,:,7].flatten())),
#          bins=4)
# plt.xlabel("Normalised Residual in dV")
# plt.savefig(master_pdir + "best-dv.pdf")

