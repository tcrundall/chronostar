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
sys.path.insert(0, '..')

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

precs = ['perf', 'half', 'gaia']
# load all the relevant data into a massive array where
# ix implicitly correspond to the value of the parameter

rdir = "../results/synth_fit/"
chain_file = "final_chain.npy"
origin_file = "origins.npy"

prec_val = {'perf':0., 'half':0.5, 'gaia':1.0}

#------------------------------------------------------------
#-------   GATHER FITS FROM THE 'ODD' SIMULATIONS -----------
#------------------------------------------------------------
logging.info("Gathering 'ODD' simulations")
o_fits = np.zeros((len(ages), len(o_spreads), len(o_v_disps), len(o_sizes),
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

#problem_ix = np.array(np.where(abs(residuals) > 10)).T

# plot all great_ixs on 2D age, pos spread histogram
#plt.clf()
#plt.hist2d(e_norm_res[e_great_ixs][:,-1].flatten(),
#           e_norm_res[e_great_ixs][:,5].flatten(), bins=5)

# PLOT ALL RESIDUALS CUT BY EACH PARAMETER
plt.clf()
plt.hist(np.hstack((o_norm_res[0:2,:,:,:,:,-1].flatten(),
                    e_norm_res[0:2,:,:,:,:,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-age-low.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[2:,:,:,:,:,-1].flatten(),
                    e_norm_res[2:,:,:,:,:,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-age-high.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[:,0,:,:,:,-1].flatten(),
                    e_norm_res[:,0,:,:,:,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-dx-low.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[:,1,:,:,:,-1].flatten(),
                    e_norm_res[:,1,:,:,:,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-dx-high.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[:,:,0,:,:,-1].flatten(),
                    e_norm_res[:,:,0,:,:,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-dv-low.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[:,:,1,:,:,-1].flatten(),
                    e_norm_res[:,:,1,:,:,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-dv-high.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[:,:,:,0,:,-1].flatten(),
                    e_norm_res[:,:,:,0,:,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-count-low.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[:,:,:,1,:,-1].flatten(),
                    e_norm_res[:,:,:,1,:,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-count-high.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[:,:,:,:,1,-1].flatten(),
                    e_norm_res[:,:,:,:,1,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-prec-half.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[:,:,:,:,2,-1].flatten(),
                    e_norm_res[:,:,:,:,2,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-prec-gaia.pdf")


# REMOVE HIGH DV AND PLOT ALL RESIDUALS CUT BY EACH PARAMETER
plt.clf()
plt.hist(np.hstack((o_norm_res[0:2,:,0,:,:,-1].flatten(),
                    e_norm_res[0:2,:,0,:,:,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-age-low-dv-low.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[2:,:,0,:,:,-1].flatten(),
                    e_norm_res[2:,:,0,:,:,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-age-high-dv-low.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[:,0,0,:,:,-1].flatten(),
                    e_norm_res[:,0,0,:,:,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-dx-low-dv-low.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[:,1,0,:,:,-1].flatten(),
                    e_norm_res[:,1,0,:,:,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-dx-high-dv-low.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[:,:,0,0,:,-1].flatten(),
                    e_norm_res[:,:,0,0,:,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-count-low-dv-low.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[:,:,0,1,:,-1].flatten(),
                    e_norm_res[:,:,0,1,:,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-count-high-dv-low.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[:,:,0,:,1,-1].flatten(),
                    e_norm_res[:,:,0,:,1,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-prec-half-dv-low.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[:,:,0,:,2,-1].flatten(),
                    e_norm_res[:,:,0,:,2,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("ages-hist-prec-gaia-dv-low.pdf")

plt.clf()
plt.hist(np.hstack((o_norm_res[:,:,:,:,:,-1].flatten(),
                    e_norm_res[:,:,:,:,:,-1].flatten())), bins=6)
plt.xlabel("Normalised residual (age)")
plt.savefig("all-ages-hist.pdf")

# PLOTTING RAW RESIDUALS AS FUNCTION OF TRUE AGE
age_fit_means = []
age_fit_stds = []

for i, age in enumerate(ages):
    age_resids = np.hstack((o_norm_res[i,:,0,:,:,-1].flatten(),
                             e_norm_res[i,:,0,:,:,-1].flatten()))
    age_fit_means.append(np.mean(age_resids))
    age_fit_stds.append(np.std(age_resids))
plt.clf()
plt.errorbar(ages, age_fit_means, yerr=age_fit_stds, fmt='b.')
plt.plot(ages, np.zeros(len(ages)), color='b', ls=':')
plt.xlabel("True age [Myr]")
plt.ylabel("Offset in age [Myr]")
plt.savefig("age-residuals-age.pdf")

# PLOTTING RAW RESIDUALS AS FUNCTION OF STAR COUNT
age_fit_means = []
age_fit_stds = []
sizes = []

for i, size in enumerate(zip(o_sizes, e_sizes)):
    age_resids = o_norm_res[:,:,0,:,i,-1]
    age_fit_means.append(np.mean(age_resids))
    age_fit_stds.append(np.std(age_resids))
    sizes.append(size[0])

    age_resids = e_norm_res[:,:,0,:,i,-1]
    age_fit_means.append(np.mean(age_resids))
    age_fit_stds.append(np.std(age_resids))
    sizes.append(size[1])

plt.clf()
plt.errorbar(sizes, age_fit_means, yerr=age_fit_stds, fmt='b.')
plt.plot(sizes, np.zeros(len(sizes)), color='b', ls=':')
plt.xlabel("Star count")
plt.ylabel("Offset in age [Myr]")
plt.savefig("age-residuals-size.pdf")

# PLOTTING RAW RESIDUALS AS FUNCTION OF PRECISION
age_fit_means = []
age_fit_stds = []
vals = []

for i, prec in enumerate(precs):
    vals.append(prec_val[prec])
    age_resids = np.hstack((o_norm_res[:,:,0,:,i,-1].flatten(),
                            e_norm_res[:,:,0,:,i,-1].flatten()))
    age_fit_means.append(np.mean(age_resids))
    age_fit_stds.append(np.std(age_resids))
plt.clf()
plt.errorbar(vals, age_fit_means, yerr=age_fit_stds, fmt='b.')
plt.plot(vals, np.zeros(len(vals)), color='b', ls=':')
plt.xlabel("Fraction of Gaia error")
plt.ylabel("Offset in age [Myr]")
plt.savefig("age-residuals-prec.pdf")

# FOR ALL LOW DV PLOT 2D HISTOGRAMS OF DX AND DV, CUT BY EACH PARAMETER
plt.clf()
plt.hist2d(np.hstack((e_norm_res[:2,:,0,:,:,6].flatten(),
                      o_norm_res[:2,:,0,:,:,6].flatten())),
           np.hstack((e_norm_res[:2,:,0,:,:,7].flatten(),
                      o_norm_res[:2,:,0,:,:,7].flatten())),
           bins=5)
plt.xlabel("Normalised Residual in dX")
plt.ylabel("Normalised Residual in dV")
plt.savefig("dx-dv-age-low.pdf")

plt.clf()
plt.hist2d(np.hstack((e_norm_res[2:,:,0,:,:,6].flatten(),
                      o_norm_res[2:,:,0,:,:,6].flatten())),
           np.hstack((e_norm_res[2:,:,0,:,:,7].flatten(),
                      o_norm_res[2:,:,0,:,:,7].flatten())),
           bins=5)
plt.xlabel("Normalised Residual in dX")
plt.ylabel("Normalised Residual in dV")
plt.savefig("dx-dv-age-high.pdf")

plt.clf()
plt.hist2d(np.hstack((e_norm_res[:,0,0,:,:,6].flatten(),
                      o_norm_res[:,0,0,:,:,6].flatten())),
           np.hstack((e_norm_res[:,0,0,:,:,7].flatten(),
                      o_norm_res[:,0,0,:,:,7].flatten())),
           bins=5)
plt.xlabel("Normalised Residual in dX")
plt.ylabel("Normalised Residual in dV")
plt.savefig("dx-dv-dx-low.pdf")

plt.clf()
plt.hist2d(np.hstack((e_norm_res[:,1,0,:,:,6].flatten(),
                      o_norm_res[:,1,0,:,:,6].flatten())),
           np.hstack((e_norm_res[:,1,0,:,:,7].flatten(),
                      o_norm_res[:,1,0,:,:,7].flatten())),
           bins=5)
plt.xlabel("Normalised Residual in dX")
plt.ylabel("Normalised Residual in dV")
plt.savefig("dx-dv-dx-high.pdf")

plt.clf()
plt.hist2d(o_norm_res[:,:,0,:,:,6].flatten(),
           o_norm_res[:,:,0,:,:,7].flatten(),
           bins=5)
plt.xlabel("Normalised Residual in dX")
plt.ylabel("Normalised Residual in dV")
plt.savefig("dx-dv-dv-low.pdf")

plt.clf()
plt.hist2d(e_norm_res[:,:,0,:,:,6].flatten(),
           e_norm_res[:,:,0,:,:,7].flatten(),
           bins=5)
plt.xlabel("Normalised Residual in dX")
plt.ylabel("Normalised Residual in dV")
plt.savefig("dx-dv-dv-high.pdf")


plt.clf()
plt.hist2d(np.hstack((e_norm_res[:,:,0,0,:,6].flatten(),
                      o_norm_res[:,:,0,0,:,6].flatten())),
           np.hstack((e_norm_res[:,:,0,0,:,7].flatten(),
                      o_norm_res[:,:,0,0,:,7].flatten())),
           bins=5)
plt.xlabel("Normalised Residual in dX")
plt.ylabel("Normalised Residual in dV")
plt.savefig("dx-dv-count-low.pdf")

plt.clf()
plt.hist2d(np.hstack((e_norm_res[:,:,0,1,:,6].flatten(),
                      o_norm_res[:,:,0,1,:,6].flatten())),
           np.hstack((e_norm_res[:,:,0,1,:,7].flatten(),
                      o_norm_res[:,:,0,1,:,7].flatten())),
           bins=5)
plt.xlabel("Normalised Residual in dX")
plt.ylabel("Normalised Residual in dV")
plt.savefig("dx-dv-count-high.pdf")

plt.clf()
plt.hist2d(np.hstack((e_norm_res[:,:,0,:,0,6].flatten(),
                      o_norm_res[:,:,0,:,0,6].flatten())),
           np.hstack((e_norm_res[:,:,0,:,0,7].flatten(),
                      o_norm_res[:,:,0,:,0,7].flatten())),
           bins=5)
plt.xlabel("Normalised Residual in dX")
plt.ylabel("Normalised Residual in dV")
plt.savefig("dx-dv-prec-perf.pdf")

plt.clf()
plt.hist2d(np.hstack((e_norm_res[:,:,0,:,1,6].flatten(),
                      o_norm_res[:,:,0,:,1,6].flatten())),
           np.hstack((e_norm_res[:,:,0,:,1,7].flatten(),
                      o_norm_res[:,:,0,:,1,7].flatten())),
           bins=5)
plt.xlabel("Normalised Residual in dX")
plt.ylabel("Normalised Residual in dV")
plt.savefig("dx-dv-prec-half.pdf")

plt.clf()
plt.hist2d(np.hstack((e_norm_res[:,:,0,:,2,6].flatten(),
                      o_norm_res[:,:,0,:,2,6].flatten())),
           np.hstack((e_norm_res[:,:,0,:,2,7].flatten(),
                      o_norm_res[:,:,0,:,2,7].flatten())),
           bins=5)
plt.xlabel("Normalised Residual in dX")
plt.ylabel("Normalised Residual in dV")
plt.savefig("dx-dv-prec-gaia.pdf")

# MAIN CONTRIBUTORS TO DX OFFSET:
# - count (higher is better)
# - dv (higher is better (that is, 1km/s vs 2km/s))
# - precision (perf is better)

plt.clf()
plt.hist(e_norm_res[:,:,0,1,:2,6].flatten(), bins=4)
plt.xlabel("Normalised Residual in dX")
plt.savefig("best-dx.pdf")

# MAIN CONTRIBUTORS TO DV OFFSET:
# - count (higher is better)
# - dx (smaller is better)

plt.clf()
plt.hist(np.hstack((o_norm_res[:,0,0,1,:,7].flatten(),
                    e_norm_res[:,0,0,1,:,7].flatten())),
         bins=4)
plt.xlabel("Normalised Residual in dV")
plt.savefig("best-dv.pdf")

