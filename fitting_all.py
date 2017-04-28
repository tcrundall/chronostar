#! /usr/bin/env python

from chronostar import groupfitter
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
                    help='[3] total number of groups to be fitted')
parser.add_argument('-r', '--background',  dest = 'r', default=1,
                    help='[1] number of groups to be fitted to background')
parser.add_argument('-n', '--noplots',  dest = 'n', action='store_true',
                    help='Set this flag if plotting anything will break things')
parser.add_argument('-l', '--local',  dest = 'l', action='store_true',
                    help='Set this flag if not running on raijin')
parser.add_argument('-i', '--infile',  dest = 'i',
		    default="results/bp_TGAS2_traceback_save.pkl",
                    help='The file of stellar tracebacks')
parser.add_argument('-d', '--debug',  dest = 'd', action='store_true',
                    help='Set this flag if debugging')
args = parser.parse_args()
burnin = int(args.b)
steps = int(args.p)
ngroups = int(args.g)
nbg_groups = int(args.r)
noplots = args.n
infile = args.i
debug  = args.d
local  = args.l          # used to set file save location

if not local:
    save_dir = '/short/kc5/'
else:
    save_dir = ''

try:        
    dummy = None
    pickle.dump(dummy, open(save_dir + "dummy.pkl",'w'))
    if local:
        print("*** Are you on Raijin? Calling fitting_all with '-l' ***\n"
              "***   or '--local' could have memory issues          ***")
except:
    print("If you're not running this on Raijin, with project kc5, call with"
          "'-l' or '--local' flag")
    raise UserWarning

npars_per_group = 14

#fixed_groups = ngroups*[None]
fixed_groups = None
best_fits    = None

# a rough timestamp to help keep logs in order
tstamp = str(int(time.time())%10000)

for nfixed in range(ngroups):
    # hardcoding only fitting one free group at a time
    nfree = 1

    # Determines if we are fitting a background group or not
    bg = (nfixed < nbg_groups)
    
    samples, pos, lnprob = groupfitter.fit_groups(
        burnin, steps, nfree, nfixed, infile,
        fixed_groups=fixed_groups, bg=bg, loc_debug=debug)

    nwalkers = np.shape(samples)[0]
    nsteps   = np.shape(samples)[1]
    npars    = np.shape(samples)[2] 
    flat_samples = np.reshape(samples, (nwalkers*nsteps, npars))
    
    cv_samples = anl.convert_samples(flat_samples, nfree, nfixed, npars)
    best_fit_true = anl.calc_best_fit(cv_samples)

    # if this is the first run, the best_fit accumulator is simply
    # set to be equal to the only best_fit so far
    best_fit = anl.calc_best_fit(flat_samples)
    if best_fits is None:
        best_fits = best_fit_true[:14]
    else:
        best_fits = np.append(best_fit_true[:14], best_fits, axis=0)
    
    # append the median of the free group to the list of fixed groups
    if fixed_groups is None:
        fixed_groups = [best_fit[:npars_per_group,0]]
    else:
        fixed_groups = [best_fit[:npars_per_group,0]] + fixed_groups

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

    # if dealing with last group, append weights to best_fits
    if nfixed == ngroups -1:
        best_fits = np.append(best_fits, best_fit_true[-(ngroups):], axis=0)
    
# Write up final results
anl.write_results(steps, ngroups, nbg_groups, best_fits, tstamp)

group_stem = "{}_{}".format(tstamp, ngroups)
pickle.dump(
    fixed_groups, open(save_dir+"results/groups_"+group_stem+".pkl",'w'))

# Go back and plot everything if desired
if not noplots:
    for nfixed in range(ngroups):
        file_stem = "{}_{}_{}".format(tstamp,nfree,nfixed)
        lnprob_pars = pickle.load(
            open(save_dir+"results/lnprob_"+file_stem+".pkl",'r'))
        anl.plot_lnprob(*lnprob_pars)
    
        corner_pars = pickle.load(
            open(save_dir+"results/corner_"+file_stem+".pkl",'r')) 
        anl.plot_corner(*corner_pars)
