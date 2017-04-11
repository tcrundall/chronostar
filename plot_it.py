#! /usr/bin/env python

import chronostar.analyser as anl
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--nfree',  dest = 'f', default=1,
                    help='[1] number of free groups')
parser.add_argument('-g', '--groups',  dest = 'g', default=1,
                    help='[1] total number of groups')
parser.add_argument('-t', '--tstamp',  dest = 't', default='nostamp',
                    help="['nostamp'] time stamp")
parser.add_argument('-l', '--local', dest = 'l', action='store_true',
                     help='Set this flag if not running on Raijin')
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
local  = args.l

if not local:
    save_dir = '/short/kc5/'
else:
    save_dir = ''

try:
    dummy = None
    pickle.dump(dummy, open(save_dir+"results/dummy.pkl", 'w'))
except:
    print("If you're not running this on Raijin, with project kc5, call with"
          "'-l' or '--local' flag")

for nfixed in range(ngroups):
    file_stem = "{}_{}_{}".format(tstamp,nfree,nfixed)
    lnprob_pars = pickle.load(
        open(save_dir+"results/lnprob_"+file_stem+".pkl",'r'))
    anl.plot_lnprob(*lnprob_pars)
    
    corner_pars = pickle.load(
        open(save_dir+"results/corner_"+file_stem+".pkl",'r')) 
    anl.plot_corner(*corner_pars)
