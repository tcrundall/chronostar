#! /usr/bin/env python

import chronostar.analyser as anl
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--nfree',  dest = 'f', default=1,
                    help='[1] number of free groups')
parser.add_argument('-g', '--groups',  dest = 'g', default=1,
                    help='[0] total number of groups')
parser.add_argument('-t', '--tstamp',  dest = 't', default='nostamp',
                    help="['nostamp'] time stamp")
#parser.add_argument('-n', '--noplots',  dest = 'n', action='store_true',
#                    help='Set this flag if running on a server')
#parser.add_argument('-i', '--infile',  dest = 'i',
#		    default="results/bp_TGAS2_traceback_save.pkl",
#                    help='Set this flag if running on a server')

# So far we are only fitting one group at a time so hardcoded this in
args = parser.parse_args()
ngroups = int(args.g)
tstamp = args.t
nfree  = int(args.f)


# #fixed_groups = ngroups*[None]
# fixed_groups = None
# best_fits    = None
# 
# # a rough timestamp to help keep logs in order
# tstamp = str(int(time.time())%10000)
# 
# for nfixed in range(ngroups):
#     # hardcoding only fitting one free group at a time
#     nfree = 1
# 
#     # Determines if we are fitting a background group or not
#     bg = (nfixed < nbg_groups)
#     
#     samples, pos, lnprob = groupfitter.fit_groups(
#         burnin, steps, nfree, nfixed, infile,
#         fixed_groups=fixed_groups, bg=bg)
# 
#     nwalkers = np.shape(samples)[0]
#     nsteps   = np.shape(samples)[1]
#     npars    = np.shape(samples)[2] 
#     flat_samples = np.reshape(samples, (nwalkers*nsteps, npars))
#     
#     cv_samples = anl.convert_samples(flat_samples, nfree, nfixed, npars)
#     best_fit_true = anl.calc_best_fit(cv_samples)
# 
#     # if this is the first run, the best_fit accumulator is simply
#     # set to be equal to the only best_fit so far
#     best_fit = anl.calc_best_fit(flat_samples)
#     if best_fits is None:
#         best_fits = best_fit_true[:14]
#     else:
#         best_fits = np.append(best_fit_true[:14], best_fits, axis=0)
#     
#     # append the median of the free group to the list of fixed groups
#     if fixed_groups is None:
#         fixed_groups = [best_fit[:npars_per_group,0]]
#     else:
#         fixed_groups = [best_fit[:npars_per_group,0]] + fixed_groups
# 
#     flat_samples = np.reshape(samples, (nwalkers*nsteps, npars))
#     flat_lnprob  = lnprob.flatten()
# 
#     # Preserving required information to plot plots at the end of script
#     # or at a later date with a different script
#     file_stem = "{}_{}_{}".format(tstamp, nfree, nfixed)
#     lnprob_pars = (lnprob, nfree, nfixed, tstamp)
#     pickle.dump(lnprob_pars, open("results/lnprob_"+file_stem+".pkl",'w'))
#     
#     weights=(nfixed+nfree > 1)
#     cv_samples = anl.convert_samples(flat_samples, nfree, nfixed, npars)
#     corner_plot_pars = (\
#         nfree, nfixed, cv_samples, lnprob,
#         True, True, False, (not bg), weights, tstamp)
#     pickle.dump(corner_plot_pars, open("results/corner_"+file_stem+".pkl",'w'))
# 
#     # if dealing with last group, append weights to best_fits
#     if nfixed == ngroups -1:
#         best_fits = np.append(best_fits, best_fit_true[-(ngroups):], axis=0)
#     
# # Write up final results
# anl.write_results(steps, ngroups, nbg_groups, best_fits, tstamp)

# Go back and plot everything if desired

for nfixed in range(ngroups):
    file_stem = "{}_{}_{}".format(tstamp,nfree,nfixed)
    lnprob_pars = pickle.load(open("results/lnprob_"+file_stem+".pkl",'r'))
    anl.plot_lnprob(*lnprob_pars)
    
    corner_pars = pickle.load(open("results/corner_"+file_stem+".pkl",'r')) 
    anl.plot_corner(*corner_pars)
