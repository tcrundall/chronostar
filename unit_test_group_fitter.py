#!/usr/bin/env python 
from chronostar.groupfitter import Group
import chronostar.groupfitter as groupfitter
#import chronostar.debuggroupfitter as groupfitter
import chronostar.fit_group as fg
import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities.iterables import multiset_permutations
from emcee.utils import MPIPool
import sys          # used for mpi things
import pickle

print("___ Testing chronostar/groupfitter.py ___")

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
parser.add_argument('-r', '--run', dest = 'r', action='store_true')
parser.add_argument('-g', '--background', dest = 'g', action='store_true')
parser.add_argument('-i', '--input', dest='infile',
                    default='results/bp_TGAS2_traceback_save.pkl')
args = parser.parse_args() 
burnin = int(args.b) 
steps = int(args.p) 
nfree = int(args.free)
nfixed = int(args.fixed)
debugging = args.d
test_run = args.r
background = args.g
infile=args.infile

# pdb.set_trace()
# testing covariance matricies and their determinants

fixed_bg_group= [-15.41, -17.22, -21.32, -4.27, -14.39, -5.83,
                 1/73.34, 1/51.61, 1/48.83,
                 1/7.20,
                 -0.21, -0.09, 0.12,
                  0.0]

if test_run:
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
                            infile=infile, pool=pool, bg=background)

    if using_mpi:
        # Close the processes
        pool.close()

    file_stem = "{}_{}_{}_{}".format(nfree, nfixed, burnin, steps)
    pickle.dump((samples, pos, lnprob), open("logs/"+file_stem+".pkl", 'w'))

    print("Closed MPI processes")


if not test_run:
#   groupfitter = groupfitter.fit_groups(
#       burnin=10, steps=10, nfixed=1, nfree=1,
#       fixed_groups = [fixed_bg_group]
#       )
    print("Testing lnprior()")
     
    lp_nfree = 1
    lp_nfixed = 1
    lp_max_age = 20
    
    # Testing lnprior
    # dummy pars are for 1 free group and 1 fixed group
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
        res = groupfitter.lnprior(good_pars[i], lp_nfree, lp_nfixed, lp_max_age)
        assert(res == 0)
    
    for pars in bad_pars:
        res = groupfitter.lnprior(pars, lp_nfree, lp_nfixed, lp_max_age)
        assert(res == -np.inf), "par: {}, res: {}".format(pars, res)
    
    free_group_npars = 15
    
    #max_nfixed = nfixed
    print("Testing generate_parameter_list()")
    for gpl_nfixed in range(0,5):
        for gpl_nfree in range(1,5):
            res_pars, res_sdev, nwalkers  = groupfitter.generate_parameter_list(
                gpl_nfixed, gpl_nfree, bg=False
                )
            res_len = len(res_pars)
            expected = free_group_npars * gpl_nfree + gpl_nfixed - 1
            assert(res_len ==  expected),\
                "*** Expected: {}, got: {}".format(expected, res_len)

    print("Testing interp_icov()")
    # comparing my implementation of interpolating with
    # output from Mike's original
    star_params = groupfitter.read_stars(infile)
    target_time = 0.5*(star_params['times'][0] + star_params['times'][1])
    mns, covs = groupfitter.interp_cov(target_time, star_params)
    icovs = np.linalg.inv(covs)
    icov_dets = np.linalg.det(icovs)

    orig_mns, orig_covs = fg.interp_cov(target_time, star_params)
    orig_icovs = np.linalg.inv(orig_covs)
    orig_icov_dets = np.linalg.det(orig_icovs)
    assert np.allclose(mns, 0.5 * (star_params['xyzuvw'][:,0] +
                                   star_params['xyzuvw'][:,1]))
    assert np.allclose(orig_mns, mns)
    assert np.allclose(orig_icovs, icovs)
    assert np.allclose(orig_icov_dets, icov_dets)

    print("Testing calc_average_eig()")
    assert(groupfitter.calc_average_eig(good_pars[0]) == 1.0)
    manual_approx_vol = np.mean(1/np.array((fixed_bg_group)[6:9]))
    res = groupfitter.calc_average_eig(fixed_bg_group)
    check = np.abs(manual_approx_vol - res)/res
    assert(check < 0.01)

    print("___ chronostar/groupfitter.py passing all tests ___")
