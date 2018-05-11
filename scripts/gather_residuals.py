from __future__ import division, print_function
"""
This script goes through a set of synthetic fit results and, using the 
final chains of the sampling along with the true data initialisation
parameters, calculates the residuals (how much I'm off by) and the
normalised residuals (if how much I think I'm off by is consistent with
how much I'm off by)
"""
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
spreads = [1, 5]
v_disps = [2, 10]
sizes = [25, 100]
precs = ['perf', 'half', 'gaia']
# load all the relevant data into a massive array where
# ix implicitly correspond to the value of the parameter

rdir = "../results/synth_fit/brightest_error/"
chain_file = "final_chain.npy"
origin_file = "origins.npy"

all_res = np.zeros((len(ages), len(spreads), len(v_disps), len(sizes),
                    len(precs), 9, 3))

for age_ix, age in enumerate(ages):
    for spread_ix, spread in enumerate(spreads):
        for v_disp_ix, v_disp in enumerate(v_disps):
            for size_ix, size in enumerate(sizes):
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
                    all_res[age_ix,spread_ix,v_disp_ix,size_ix,prec_ix]=\
                        key_info

residuals = (all_res[:,:,:,:,:,:,0]-all_res[:,:,:,:,:,:,2])
normed_residuals = residuals / all_res[:,:,:,:,:,:,1]

# the fits where all parameters are within 15 sigma
worst_ixs = np.where(abs(normed_residuals).max(axis=-1) > 15)
bad_ixs = np.where(abs(normed_residuals).max(axis=-1) > 5)

fine_ixs = np.where(abs(normed_residuals).max(axis=-1) < 15)
great_ixs = np.where(abs(normed_residuals).max(axis=-1) < 5)

#problem_ix = np.array(np.where(abs(residuals) > 10)).T

# plot all great_ixs on 2D age, pos spread histogram
plt.clf()
plt.hist2d(normed_residuals[great_ixs][:,-1].flatten(),
           normed_residuals[great_ixs][:,5].flatten(), bins=20)

