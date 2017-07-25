#! /usr/bin/env python
"""Use traced back data to find best fitting groups.
"""
import chronostar.analyser as anl
from chronostar import groupfitter
import numpy as np
import time
import pickle
import pdb
import argparse
import sys
from emcee.utils import MPIPool
import matplotlib.pyplot as plt

# ---------------------------------
# --   SETTING EVERYTHING UP     --
# ---------------------------------

#A bunch of arguments determing how to run the emcee
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--burnin', dest = 'b', default=10,
                    help='[10] number of burn-in steps')
parser.add_argument('-p', '--nsteps',  dest = 'p', default=20,
                    help='[20] number of sampling steps')
parser.add_argument('-f', '--free',  dest = 'f', default=1,
                    help='[1] number of free groups to be fitted')
parser.add_argument('-r', '--background',  dest = 'r', default=0,
                    help='[0] number of groups to be fitted to background'+
                    ' i.e. fixed age of 0')
# parser.add_argument('-a', '--age',  dest = 'a', default=None,
#                     help='[None] fixed age for all free groups')
parser.add_argument('-n', '--noplots',  dest = 'n', action='store_true',
                    help='Set this flag if plotting anything will break things')
parser.add_argument('-l', '--local',  dest = 'l', action='store_true',
                    help='Set this flag if not running on raijin')
parser.add_argument('-i', '--infile',  dest = 'i',
                    default='data/BPMG_traceback_165Myr.pkl',
                    help='The file of stellar tracebacks')
parser.add_argument('-d', '--debug',  dest = 'd', action='store_true',
                    help='Set this flag if debugging')

# Setting up mpi
using_mpi = True
try:
    # Initialize the MPI-based pool used for parallelization.
    pool = MPIPool()
except:
    print("MPI doesn't seem to be installed... maybe install it?")
    print("MPI available! - call this with e.g. mpirun -np 4 python"+
          " fit_groups.py")
    using_mpi = False
    pool=None

if using_mpi:
    if not pool.is_master():
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)
    else:
        print("MPI available! - call this with e.g. mpirun -np 4 python"+
              " fit_grups.py")

# interpret argument flags and values
args    = parser.parse_args()
burnin  = int(args.b)     # emcee pars
nsteps   = int(args.p)     # emcee pars
nfree   = int(args.f)     # number of free groups
nfixed  = int(args.r)     # number of background groups (age=0)
noplots = args.n
infile  = args.i
debug   = args.d
local   = args.l          # used to choose file save location

# Making file saving generic w.r.t. rsaa servers or raijin
if not local:
    data_dir    = '/short/kc5/data/'
    results_dir = '/short/kc5/results/' 
else: 
    data_dir    = 'data/' 
    results_dir = 'results/' 
    #save_dir = '' 
 
try:
    open(data_dir + "temp.txt",'w')
except:
    print("*** If you're not running this on Raijin, with project kc5, ***\n"+
          "*** call with '-l' or '--local' flag                        ***")
    sys.exit()
    # raise UserWarning

# a rough timestamp to help keep logs in order
# if fixed_ages:
#    tstamp = str(fixed_age).replace('.','_')\
#               + "_" + str(int(time.time())%1000000)
# else:
tstamp = str(int(time.time())%1000000)
print("Time stamp is: {}".format(tstamp))


# --------------------------------
# --         RUNNING FIT        --
# --------------------------------

bg = False
samples, pos, lnprob =\
    groupfitter.fit_groups(
        burnin, nsteps, nfree, nfixed, infile=infile,
        bg=bg, loc_debug=debug)
print("Run finished")

# --------------------------------
# --      ANALYSING RESULT      --
# --------------------------------

# get the number of parameters used from shape of samples array
# samples.shape -> (nwalkers, nsteps, npars)
nwalkers = np.shape(samples)[0]
npars = np.shape(samples)[2]

# reshape samples array in order to combine each walker's path into one chain
flat_samples = np.reshape(samples, (nwalkers*nsteps, npars))
# flat_lnprob  = lnprob.flatten() # no idea why I even need this...

# calculate the volume of each error ellipse
vols =  np.zeros(flat_samples.shape[0])
for i in range(vols.shape[0]):
    vols[i] = groupfitter.calc_average_eig(flat_samples[i])
best_vol = anl.calc_best_fit(vols.reshape(-1,1))

# converts the sample pars into physical pars, e.g. the inverse of stdev
# is explored, the weight of the final group is implicitly derived
cv_samples = anl.convert_samples(flat_samples, nfree, nfixed, npars)

# Calculate the median, positive error and negative error values
best_fits = anl.calc_best_fit(cv_samples)

# Generate log written into local logs/ directory
anl.write_results(nsteps, nfree, nfixed, best_fits, tstamp, 
                  bw=best_vol, infile=infile)
print("Logs written")

# --------------------------------
# --      PLOTTING RESULT       --
# --------------------------------

# In order to keep things generic w.r.t. raijin, the plotting data is stored
# and a separate script is provided to come back and plot later
file_stem = "{}_{}_{}".format(nfree, nfixed, nsteps)
lnprob_pars = (lnprob, nfree, nfixed, tstamp)
pickle.dump(
    lnprob_pars,
    open(results_dir+"{}_lnprob_".format(tstamp)+file_stem+".pkl",'w'))

# Only include weights in corner plot if there are more than 1 group being
# fitted. corner complains when asked to plot a fixed parameter and 1 group
# will have a fixed weight of 1.
weights=(nfixed+nfree > 1)
cv_samples = anl.convert_samples(flat_samples, nfree, nfixed, npars)
corner_plot_pars = (
    nfree, nfixed, cv_samples, lnprob,
    True, True, False, (not bg), weights, tstamp)
pickle.dump(
    corner_plot_pars,
    open(results_dir+"{}_corner_".format(tstamp)+file_stem+".pkl",'w') )

print("Plot data saved. Go back and plot with:\n"+
      "./plot_it.py -t {} -f {} -r {} -p {} -l\n".\
          format(tstamp, nfree, nfixed, nsteps)+
      "   if not on raijin, or:\n"+
      "./plot_it.py -t {} -f {} -r {} -p {}\n".\
          format(tstamp, nfree, nfixed, nsteps))

if using_mpi:
    pool.close()
