#! usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
sys.path.insert(0, '..')

import chronostar.retired.tfgroupfitter as tfgf
import chronostar.retired.synthesiser as syn
import chronostar.retired.tracingback as tb
import chronostar.transform as tf
import chronostar.retired.error_ellipse as ee
from chronostar.retired import utils

dX = 10.
dY = 10.
dZ = 10.
dXav_check = (dX*dY*dZ)**(1./3.)
dV = 5.
age = 10
nstars = 100

# vanilla sphere
# group_pars_old_style = [50,-20,0,0,1,0,dX,dX,dX,dV,0.,0.,0., age, nstars]

# same volume, distorted spatially
group_pars_old_style =\
    [50,-20,0,0,5,0,dX,dY,dZ,dV,0.,0.,0., age, nstars]
then_cov_true = utils.generate_cov(utils.internalise_pars(
    group_pars_old_style
))

dXav = (np.prod(np.linalg.eigvals(then_cov_true[:3,:3]))**(1./6.))

# This represents the target result - a simplified, spherical starting point
group_pars_tf_style =\
    np.append(
        np.append(
            np.append(np.copy(group_pars_old_style)[:6], dXav), dV
        ), age
    )

tb_file = 'temp_traceforward_tb.pkl'
synth_file = 'temp_traceforward_synth.pkl'
#error = 1e-5

error = 0.5

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
burnin_steps = 500
sampling_steps = 500
print("Burning in: {}".format(burnin_steps))
print("Sampling: {}".format(sampling_steps))
best_fit, sampler_chain, lnprob = tfgf.fit_group(
    tb_file, burnin_steps=burnin_steps, sampling_steps=sampling_steps,
    plot_it=True
)
means = best_fit[0:6]
stds = 1 / best_fit[6:8]
fitted_age = best_fit[8]

tol_mean = 3.5
tol_std = 2.5
tol_age = 0.5



# Plot the origin point, current stars, and fit to origin

# plotting stars
star_pars = tfgf.read_stars(tb_file=tb_file)
xyzuvw = star_pars['xyzuvw']

then_cov_true = utils.generate_cov(utils.internalise_pars(group_pars_old_style))
then_cov_simple = tfgf.generate_cov(group_pars_in)
then_cov_fitted = tfgf.generate_cov(best_fit)
now_cov_fitted = tf.transformCovMat(then_cov_fitted, tb.trace_forward,
                                  best_fit[0:6], dim=6, args=(best_fit[-1],))
now_mean_fitted = tb.trace_forward(best_fit[:6], best_fit[-1])

plt.clf()


def plot_results():
    plt.plot(xyzuvw[:,0,0], xyzuvw[:,0,1], 'b.')
    ee.plot_cov_ellipse(then_cov_simple[:2,:2], group_pars_tf_style[:2], color='orange',
                        alpha=0.2, hatch='|', ls='--')
    ee.plot_cov_ellipse(then_cov_true[:2,:2], group_pars_tf_style[:2], color='orange',
                        alpha=1, ls = ':', fill=False)
    ee.plot_cov_ellipse(then_cov_fitted[:2,:2], best_fit[:2], color='xkcd:neon purple',
                        alpha=0.2, hatch='/', ls='-.')
    ee.plot_cov_ellipse(now_cov_fitted[:2,:2], now_mean_fitted[:2], color='b',
                        alpha=0.03, hatch='.')

    plt.savefig("temp_tf_fits.png")

plot_results()

if True:
    assert np.max(abs(means - group_pars_tf_style[0:6])) < tol_mean,\
       "\nReceived:\n{}\nshould be within {} to:\n{}".\
            format(means, tol_mean, group_pars_tf_style[0:6])
    assert np.max(abs(stds - group_pars_tf_style[6:8])) < tol_std,\
        "\nReceived:\n{}\nshould be close to:\n{}".\
            format(stds, tol_std, group_pars_tf_style[6:8])
    assert np.max(abs(fitted_age - group_pars_tf_style[8])) < tol_age,\
        "\nReceived:\n{}\nshould be close to:\n{}".\
        format(fitted_age, tol_age, group_pars_tf_style[8])

