import numpy as np
import pickle
import matplotlib.pyplot as plt
import corner
import pdb
import argparse
import sys

"""
    x : array-like
        The group parameters, which are...
        GROUP 1:
        x[0] to x[5] : xyzuvw
        x[6] to x[8] : positional variances in x,y,z
        x[9]  : velocity dispersion (symmetrical for u,v,w)
        x[10] to x[12] :  correlations between x,y,z
        x[13] : birth time of group in Myr
        x[14] : fraction of stars in group 1
        GROUP 2:
        x[15] to x[20] : xyzuvw
        x[21] to x[23] : positional variances in x,y,z
        x[24]  : velocity dispersion (symmetrical for u,v,w)
        x[25] to x[27] :  correlations between x,y,z
        x[28] : birth time of group in Myr
        x[29] : fraction of stars in group 2
        BACKGROUND:
        x[30] to x[35] : xyzuvw of background (BG)
        x[36] to x[38] : positional variances in x,y,z of BG
        x[39]  : velocity dispersion (symmetrical for u,v,w) of BG
        x[40] to x[42] :  correlations between x,y,z of BG
"""

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--steps',  dest = 'p', default=10000,
                                    help='[1000] number of sampling steps')
parser.add_argument('-b', '--burnin', dest = 'b', default=2000,
                                    help='[700] number of burn-in steps')

#min_lnprob = -10672
args = parser.parse_args()
steps = int(args.p)
burn_in = int(args.b)
dims = 43

pdb.set_trace()
samples, lnprob = pickle.load(open("logs/bp_three_{}_{}.pkl".format(steps,burn_in)))
flat_samples = np.reshape(samples,(-1,dims))
flat_lnprob  = np.reshape(lnprob, (-1,))
pdb.set_trace()

best_ix = np.argmax(flat_lnprob)
best_sample = flat_samples[best_ix]

# below two lines are problematic
#min_lnprob = np.percentile(lnprob, 95)
#best_samples = samples[np.where(lnprob>min_lnprob)]

#mn = np.linspace(0,dims-1,dims)
#for i in range(dims):
#    mn[i] = np.mean(samples[np.where(lnprob > min_lnprob)][:,i])

wanted_pars = [6,7,8,9,13,14,21,22,23,24,28,29]
labels=["dX1", "dY1", "dZ1", "dV1", "age1", "weight1", "dX2", "dY2", "dZ2", "dV2", "age2", "weight2"]

# is broken... somehow.... I think it complains because the selected
# samples don't vary enough
#resamples = np.resize(best_samples,(-1,dims))
pdb.set_trace()
fig = corner.corner(flat_samples[:,wanted_pars], truths=best_sample[wanted_pars], labels=labels)
fig.savefig("plots/bp_three_{}_{}_best.png".format(steps, burn_in))

pdb.set_trace()
