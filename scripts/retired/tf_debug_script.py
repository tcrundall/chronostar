#! /usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0,'../../..')

import chronostar.tfgroupfitter as tfgf
import chronostar.transform as tf
import chronostar.traceback as tb
import chronostar.error_ellipse as ee
from chronostar import utils

results = []
for i in range(30):
    try:
        res = np.load("burnin_chain{}.npy".format(i))
        print("Successfully loaded chain {}".format(i))
        results.append(res)
    except IOError:
        pass

final = np.load("result.npy")
star_pars = tfgf.read_stars("tb_data.pkl")

print("Origin's score: {}".format(tfgf.lnprobfunc(final[3], star_pars)))
print("Fitted's score: {}".format(tfgf.lnprobfunc(final[0], star_pars)))


nstars = star_pars['xyzuvw'].shape[0]
star_covs = star_pars['xyzuvw_cov'][:,0]
star_mns  = star_pars['xyzuvw'][:,0]
nstars = star_mns.shape[0]

fitted_mean_then = final[0][:6]
fitted_cov_then = tfgf.generate_cov(final[0])
fitted_age = final[0][8]

fitted_mean_now = tb.trace_forward(fitted_mean_then, fitted_age)
fitted_cov_now = tf.transform_cov(
    fitted_cov_then, tb.trace_forward, fitted_mean_then, dim=6,
    args=(fitted_age,)
)
lnols = tfgf.get_lnoverlaps(
    fitted_cov_now, fitted_mean_now, star_covs, star_mns, nstars
)

origin_mean_then = final[3][:6]
origin_cov_then = tfgf.generate_cov(final[3])
origin_age = final[3][8]

origin_mean_now = tb.trace_forward(origin_mean_then, origin_age)
origin_cov_now = tf.transform_cov(
    origin_cov_then, tb.trace_forward, origin_mean_then, dim=6,
    args=(origin_age,)
)
o_lnols = tfgf.get_lnoverlaps(
    origin_cov_now, origin_mean_now, star_covs, star_mns, nstars
)

true_orign_cov_then = utils.generate_cov(utils.internalise_pars(final[5]))
