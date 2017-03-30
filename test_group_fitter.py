#!/usr/bin/env python

from chronostar.groupfitter import Group
import chronostar.groupfitter as groupfitter
import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities.iterables import multiset_permutations
from emcee.utils import MPIPool
import sys          # used for mpi things
import pickle

params = [-6.574, 66.560, 23.436, -1.327,-11.427, -6.527,\
        10.045, 10.319, 12.334,  0.762,  0.932,  0.735,  0.846]
age = 20.589

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--burnin', dest = 'b', default=200,
                                    help='[200] number of burn-in steps')
parser.add_argument('-p', '--steps',  dest = 'p', default=100,
                                    help='[100] number of sampling steps')
parser.add_argument('-f', '--free', dest = 'free', default=1,
                                    help='[1] number of free groups')
parser.add_argument('-x', '--fixed', dest = 'fixed', default=1,
                                    help='[1] number of fixed groups')
parser.add_argument('-d', '--debug', dest = 'd', action='store_true')
parser.add_argument('-t', '--test', dest = 't', action='store_true')
parser.add_argument('-i', '--input', dest='infile', default='results/bp_TGAS2_traceback_save.pkl')
args = parser.parse_args() 
burnin = int(args.b) 
steps = int(args.p) 
nfree = int(args.free)
nfixed = int(args.fixed)
debugging = args.d
test_run  = args.t
infile=args.infile

# pdb.set_trace()
# testing covariance matricies and their determinants

fixed_bg_group= [-15.41, -17.22, -21.32, -4.27, -14.39, -5.83,
                 1/73.34, 1/51.61, 1/48.83,
                 1/7.20,
                 -0.21, -0.09, 0.12,
                  0.0]

print("Initialising myFitter...")
if not test_run:
    print("About to set up mpi")

    using_mpi = True
    try:
        # Initialize the MPI-based pool used for parallelization.
        pool = MPIPool()
    except:
        print("Either MPI doesn't seem to be installed or you aren't"
              "running with MPI... ")
        using_mpi = False
        pool=None

    if using_mpi:
        if not pool.is_master():
            # Wait for instructions from the master process.
            pool.wait()
            sys.exit(0)
    else:
        print("MPI available for this code! - call this with"
              "e.g. mpirun -np 16 python test_fitthree_bp.py")

    samples, pos, lnprob = groupfitter.fit_groups(
                            burnin=burnin, steps=steps, nfixed=nfixed,
                            nfree=nfree, fixed_groups=nfixed*[fixed_bg_group],
                            infile=infile, pool=pool)

    if using_mpi:
        # Close the processes
        pool.close()

    file_stem = "{}_{}_{}_{}".format(nfree, nfixed, burnin, steps)
    pickle.dump((samples, pos, lnprob), open("logs/"+file_stem+".pkl", 'w'))

    print("Closed MPI processes")


if (test_run):
    myTestFitter = GroupFitter(burnin=10, steps=10, nfixed=1, nfree=1,
                        fixed_groups = [fixed_bg_group])
    print("About to test lnprior")
    
    # Testing lnprior
    good_pars = np.array([
                  [0,0,0,0,0,0, 1,1,1,1, 0,0,0, 10, 0.5],
                  [10,0,0,0,0,0, 1,1,1,1, 0,0,0, 10, 0.5],
                  [10,10,10,10,10,10, 1,1,1,1, 0.5,0.5,0.5, 15, 0.5],
                  [20,10,10,10,10,10, 1,1,1,1, -0.5,0.5,0.5, 15, 0.5]
                ])
    
    bad_pars = [ [0,0,0,0,0,0,-1,1,1,1,0,0,0,10,0.5], # negative dX
                  [10,10,10,10,10,10,1,1,1,1,1.5,0.5,0.5, 15, 0.5],# xycorr > 1
                  [0,0,0,0,0,0,1,1,1,1, 0,0,0,10,1.5], # amplitude > 1
                  [10,10,10,10,10,10,1,1,1,1,0.5,0.5,0.5,-15,0.5]# negative age
                 ]
    
    #pdb.set_trace()
    for i in range(2):
        assert(myTestFitter.lnprior(good_pars[i]) == 0)
    
    
    for pars in bad_pars:
        assert(myTestFitter.lnprior(pars) == -np.inf), "par: {}".format(pars)
    
    print("lnprior seems fine...")

    free_group_npars = 15
    
    max_nfixed = nfixed
    print("Testing generate_parameter_list()")
    for i in range(1,3):
        for nfree in range(1,3):
            nfixed = i
            print("-- {} fixed and {} free".format(nfixed, nfree))
            res_pars, res_sdev = myTestFitter.\
                                    generate_parameter_list(nfixed,nfree,
                                                            bg=False)
            res_len = len(res_pars)
            if nfixed > max_nfixed:
                nfixed = max_nfixed
            expected = free_group_npars * nfree + nfixed - 1
            assert(res_len ==  expected), "*** Expected: {}, got: {}".format(expected, res_len)
    
    if not debugging:
        # result could be filename of samples
        result = myTestFitter.fit_groups(nfixed, 1) 

