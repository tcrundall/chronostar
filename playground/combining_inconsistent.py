from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sig):
    amp = 1 / np.sqrt(2. * np.pi * np.power(sig, 2.))
    return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def combine_sigs(sig1, sig2):
    return 0.5*np.sqrt(sig1**2 + sig2**2)

nstars = 3
gaia_rvs = np.array([5.]*nstars)#, 10., 15., 20., 15.])
gaia_ervs = np.array([0.2]*nstars)

other_rvs = np.array([5., 7., 10., 13.])
other_ervs = np.array([0.4]*nstars)

plt.clf()
colors = 'brgy'
ngaia_pts = 1000
gaia_pts = np.random.randn(ngaia_pts)*gaia_ervs[0] + gaia_rvs[0]
xs = np.linspace(3,12,1000)
plt.plot(xs, gaussian(xs, gaia_rvs[0], gaia_ervs[0]), c='orange',
         label="Gaia")
for i in range(nstars):
    plt.plot(xs, gaussian(xs, other_rvs[i], other_ervs[i]), c=colors[i], ls='-',
             label="Other {}".format(i))
    comb_mu = (gaia_rvs[i] + other_rvs[i])*0.5
    comb_sig = combine_sigs(gaia_ervs[i], other_ervs[i])
    plt.plot(xs, gaussian(xs, comb_mu, comb_sig), c=colors[i], ls='--',
             label="Classic comb {}".format(i))

    nother_pts = 100
    other_pts = np.random.randn(nother_pts)*other_ervs[i] + other_rvs[i]
    mc_mu = np.mean(np.append(gaia_pts, other_pts))
    mc_sig = np.std(np.append(gaia_pts, other_pts))

    plt.plot(xs, gaussian(xs, mc_mu, mc_sig), c=colors[i], ls='-.',
             label="MC comb {}".format(i))

plt.legend()



plt.savefig("temp_comb_inc.pdf")