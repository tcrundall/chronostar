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
                    help='Set this flag if running on a server')
args = parser.parse_args()
burnin = int(args.b)
steps = int(args.p)
ngroups = int(args.g)
nbg_groups = int(args.r)
noplots = args.n

infile = "results/bp_TGAS2_traceback_save.pkl"
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
        fixed_groups=fixed_groups, bg=bg)

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
    pickle.dump(lnprob_pars, open("results/lnprob_"+file_stem+".pkl",'w'))
    
    weights=(nfixed+nfree > 1)
    cv_samples = anl.convert_samples(flat_samples, nfree, nfixed, npars)
    corner_plot_pars = (\
        nfree, nfixed, cv_samples, lnprob,
        True, True, False, (not bg), weights, tstamp)
    pickle.dump(corner_plot_pars, open("results/corner_"+file_stem+".pkl",'w'))

    # if dealing with last group, append weights to best_fits
    if nfixed == ngroups -1:
        best_fits = np.append(best_fits, best_fit_true[-(ngroups):], axis=0)
    
# Write up final results
anl.write_results(steps, ngroups, nbg_groups, best_fits, tstamp)

# Go back and plot everything if desired
if not noplots:
    for nfixed in range(ngroups):
        file_stem = "{}_{}_{}".format(tstamp,nfree,nfixed)
        lnprob_pars = pickle.load(open("results/lnprob_"+file_stem+".pkl",'r'))
        anl.plot_lnprob(*lnprob_pars)
    
        corner_pars = pickle.load(open("results/corner_"+file_stem+".pkl",'r')) 
        anl.plot_corner(*corner_pars)
