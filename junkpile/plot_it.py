#! /usr/bin/env python
"""Script that auto generates all the plots from snapshot data stored in 
pkl files. Need to provide a timestamp and other parameters required
to identify the file names.
"""

import chronostar.retired.analyser as anl
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--nfree',  dest = 'f', default=1,
                    help='[1] number of free groups')
parser.add_argument('-r', '--background',  dest = 'r', default=0,
                    help='[1] number of groups fitted to background'+
                         ' i.e. fixed age of 0')
parser.add_argument('-t', '--tstamp',  dest = 't', default='nostamp',
                    help="['nostamp'] time stamp")
parser.add_argument('-p', '--nsteps',  dest = 'p', default=20,
                    help='[20] number of steps')
parser.add_argument('-l', '--local', dest = 'l', action='store_true',
                     help='Set this flag if not running on Raijin')
# parser.add_argument('-r', '--range', dest = 'r', action='store_true')
# parser.add_argument('-c', '--corner', dest = 'c', action='store_true')

# So far we are only fitting one group at a time so hardcoded this in
args = parser.parse_args()
tstamp = args.t
nfree  = int(args.f)
nfixed = int(args.r)
nsteps = int(args.p)
local  = args.l
# corner = args.c
# span_range = args.r

if local:
    results_dir = 'results/'
else:
    results_dir = '/short/kc5/results/'

try:
    dummy = None
    pickle.dump(dummy, open(results_dir+"dummy.pkl", 'w'))
except:
    print("If you're not running this on Raijin, with project kc5, call with"
          "'-l' or '--local' flag")

# if span_range:
#     for nfixed in range(ngroups):
#         file_stem = "{}_{}_{}".format(tstamp,nfree,nfixed)
#         if not corner:
#             lnprob_pars = pickle.load(
#                 open(results_dir+"lnprob_"+file_stem+".pkl",'r'))
#             anl.plot_lnprob(*lnprob_pars)
#         
#         corner_pars = pickle.load(
#             open(results_dir+"corner_"+file_stem+".pkl",'r')) 
#         anl.plot_corner(*corner_pars)

# need to be generic with legacy format
try:
    file_stem = "{}_{}_{}".format(nfree,nfixed,nsteps)
    
    # plotting lnprob
    lnprob_pars = pickle.load(
        open(results_dir+"{}_lnprob_".format(tstamp)
                       +file_stem+".pkl",'r'))
    anl.plot_lnprob(*lnprob_pars)
    
    # plotting corner
    corner_pars = pickle.load(
        open(results_dir+"{}_corner_".format(tstamp)
        +file_stem+".pkl",'r')) 
    anl.plot_corner(*corner_pars)
except:
    # LEGACY FORMAT
    file_stem = "{}_{}_{}_{}".format(tstamp,nfree,nfixed,nsteps)
    lnprob_pars = pickle.load(
        open(results_dir+"lnprob_"+file_stem+".pkl",'r'))
    anl.plot_lnprob(*lnprob_pars)

    corner_pars = pickle.load(
        open(results_dir+"corner_"+file_stem+".pkl",'r')) 
    anl.plot_corner(*corner_pars)

