#! /usr/bin/env python
# use for debugging age problem:
import chronostar.analyser as anl
import numpy as np
import time
import pickle

import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--burnin', dest = 'b', default=10,
                    help='[10] number of burn-in steps')
parser.add_argument('-p', '--steps',  dest = 'p', default=20,
                    help='[20] number of sampling steps')
parser.add_argument('-g', '--groups',  dest = 'g', default=3,
                    help='[1] total number of groups to be fitted')
parser.add_argument('-r', '--background',  dest = 'r', default=0,
                    help='[0] number of groups to be fitted to background')
parser.add_argument('-n', '--noplots',  dest = 'n', action='store_true',
                    help='Set this flag if plotting anything will break things')
parser.add_argument('-l', '--local',  dest = 'l', action='store_true',
                    help='Set this flag if not running on raijin')
parser.add_argument('-i', '--infile',  dest = 'i',
		    default="TWA_traceback_20Myr.pkl",
                    help='The file of stellar tracebacks')
parser.add_argument('-d', '--debug',  dest = 'd', action='store_true',
                    help='Set this flag if debugging')
parser.add_argument('-a', '--age',  dest = 'a', action='store_true',
                    help='Set this flag if debugging age issue')
args = parser.parse_args()
burnin = int(args.b)
steps = int(args.p)
ngroups = int(args.g)
nbg_groups = int(args.r)
noplots = args.n
infile = args.i
debug  = args.d
local  = args.l          # used to set file save location
debug_age = args.a

if debug_age:
    from chronostar import debuggroupfitter as groupfitter
    if burnin < 200:
        burnin = 200
else:
    from chronostar import groupfitter

if not local:
    save_dir = '/short/kc5/'
else:
    save_dir = 'data/'

try:        
    dummy = None
    pickle.dump(dummy, open(save_dir + "dummy.pkl",'w'))
    if local:
        print("*** Are you on Raijin? Calling fitting_bp with '-l'  ***\n"
              "***   or '--local' could have memory issues          ***")
except:
    print("If you're not running this on Raijin, with project kc5, call with"
          "'-l' or '--local' flag")
    raise UserWarning

group_names, initial_groups, ages_all = pickle.load(
    open(save_dir + "init_mgs.pkl", 'r'))
#pdb.set_trace()

#twa_init_xyzuvw = [initial_groups[1]]
#twa_init_age    = [ages_all[1]]

npars_per_group = 14

#fixed_groups = ngroups*[None]
# fixed groups is a pickle dump of a previous run of fitting just bg
#fixed_groups = pickle.load(open("/short/kc5/results/groups_5479_1_2.pkl", 'r'))

#fixed_groups = pickle.load(open("/short/kc5/results/groups_3409_10.pkl", 'r'))
#best_fits    = None

# a rough timestamp to help keep logs in order
tstamp = str(int(time.time())%10000)
print("Time stamp is: {}".format(tstamp))

# hardcoding only fitting one free group at a time
nfree = 1
#nfixed = 3
nfixed = 0
ngroups = nfree + nfixed

bg = False

samples, pos, lnprob = groupfitter.fit_groups(
    burnin, steps, nfree, nfixed, infile,
    bg=bg, loc_debug=debug)

nwalkers = np.shape(samples)[0]
nsteps   = np.shape(samples)[1]
npars    = np.shape(samples)[2] 
flat_samples = np.reshape(samples, (nwalkers*nsteps, npars))

cv_samples = anl.convert_samples(flat_samples, nfree, nfixed, npars)
best_fits = anl.calc_best_fit(cv_samples)

flat_samples = np.reshape(samples, (nwalkers*nsteps, npars))
flat_lnprob  = lnprob.flatten()

# Preserving required information to plot plots at the end of script
# or at a later date with a different script
file_stem = "{}_{}_{}".format(tstamp, nfree, nfixed)
lnprob_pars = (lnprob, nfree, nfixed, tstamp)
pickle.dump(
    lnprob_pars, open(save_dir+"results/lnprob_"+file_stem+".pkl",'w'))

weights=(nfixed+nfree > 1)
cv_samples = anl.convert_samples(flat_samples, nfree, nfixed, npars)
corner_plot_pars = (
    nfree, nfixed, cv_samples, lnprob,
    True, True, False, (not bg), weights, tstamp)
pickle.dump(
    corner_plot_pars,
    open(save_dir+"results/corner_"+file_stem+".pkl",'w') )

# Write up final results
anl.write_results(steps, nfree, nbg_groups, best_fits, tstamp, nfixed)

#pickle.dump(fixed_groups, open(save_dir+"results/groups_"+file_stem+".pkl",'w'))

# Go back and plot everything if desired
if not noplots:
    nfixed = 0
    file_stem = "{}_{}_{}".format(tstamp,nfree,nfixed)
    lnprob_pars = pickle.load(
        open(save_dir+"results/lnprob_"+file_stem+".pkl",'r'))
    anl.plot_lnprob(*lnprob_pars)
    # pdb.set_trace()

    anl.plot_corner(
        *corner_plot_pars[0:4], ages=True, means=True, tstamp=tstamp
        )

#    corner_pars = pickle.load(
#        open(save_dir+"results/corner_"+file_stem+".pkl",'r')) 
#    anl.plot_corner(*corner_pars)

