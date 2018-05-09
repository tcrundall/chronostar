from __future__ import division, print_function
"""
This script goes through a set of synthetic fit results and, using the 
final chains of the sampling along with the true data initialisation
parameters, calculates the residuals (how much I'm off by) and the
normalised residuals (if how much I think I'm off by is consistent with
how much I'm off by)
"""

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
                     zip(*np.percentile(flat_samples, [16,50,84], axis=0))) )

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
ages = [5, 10, 20]
spreads = [2, 5, 10]
v_disps = [1, 2, 5]
sizes = [25, 50, 100]
precs = ['perf', 'half', 'gaia', 'double']
"""
ages = [5, 10]
spreads = [2]
v_disps = [1]
sizes   = [25]
precs = ['perf']
"""
# load all the relevant data into a massive array where
# ix implicitly correspond to the value of the parameter

res_dir = "../results/from_server/tf_results/"
res_file = "result.npy"

res_arr = np.zeros((len(ages), len(spreads), len(v_disps), len(sizes),
                    len(precs), 3))

for age_ix, age in enumerate(ages):
    for spread_ix, spread in enumerate(spreads):
        for v_disp_ix, v_disp in enumerate(v_disps):
            for size_ix, size in enumerate(sizes):
                for prec_ix, prec in enumerate(precs):
                    pdir = "{}_{}_{}_{}/{}/".format(
                        age, spread, v_disp, size, prec
                    )
                    result = np.load(res_dir + pdir + res_file)
                    try:
                        info = getInfo(result, -1)
                    except IndexError:
                        print("No file for: {} {} {} {} {}".format(
                            age, spread, v_disp, size, prec
                        ))
                        info = np.array([None, None, None])
                    res_arr[age_ix,spread_ix,v_disp_ix,size_ix,prec_ix] = info

residuals = (res_arr[:,:,:,:,:,0]-res_arr[:,:,:,:,:,2])

problem_ix = np.array(np.where(abs(residuals) > 10)).T




