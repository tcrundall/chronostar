#!/usr/bin/env python

from chronostar.groupfitter import MVGaussian
from chronostar.groupfitter import GroupFitter
from chronostar.groupfitter import Group
import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities.iterables import multiset_permutations
from emcee.utils import MPIPool

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
    myFitter = GroupFitter(burnin=burnin, steps=steps, nfixed=nfixed, infile=infile,
                           nfree=nfree, fixed_groups=nfixed*[fixed_bg_group])
    # result could be filename of samples
    result = myFitter.fit_groups(nfixed, nfree, pool=pool)

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

    # testing group_metric
    print("Testing group_metric") 
    assert(myTestFitter.group_metric(good_pars[0], good_pars[0]) == 0.0),\
            myTestFitter.group_metric(good_pars[0], good_pars[0])    
    assert(myTestFitter.group_metric(good_pars[1], good_pars[1]) == 0.0),\
            myTestFitter.group_metric(good_pars[1], good_pars[1])
    ngood_pars = 4
    results = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            results[i,j] = myTestFitter.group_metric(good_pars[i],
                                                     good_pars[j])

    # checking triangle inequality
    for i in range(4):
        for j in range(i,4):
            for k in range(4):
                if (k != j and k != i):
                    assert (results[i,j] <= results[i,k] + results[k,j]),\
                            "i: {}, j: {}, k: {}\n".format(i,j,k) + \
                            "{} , {}, {}".format(results[i,j], results[i,k],
                                                 results[k,j])

    #myAnalyser = SamplerAnalyser(result) # I could initialise an Analyser
                                          # object to investigate the 
                                          # sample/produce plots etc. It would
                                          #  be useful to have them separate so
                                          # I can run a bunch of runs automated
                                          # and investigate afterwards

    
    #myAnalyser.makePlots(show=True)
    #myAnalyser.write
    
    free_group_npars = 15
    
    max_nfixed = nfixed
    print("Testing generate_parameter_list()")
    for i in range(1,3):
        for nfree in range(1,3):
            nfixed = i
            print("-- {} fixed and {} free".format(nfixed, nfree))
            res_pars, res_sdev = myTestFitter.\
                                    generate_parameter_list(nfixed,nfree)
            res_len = len(res_pars)
            if nfixed > max_nfixed:
                nfixed = max_nfixed
            expected = free_group_npars * nfree + nfixed - 1
            assert(res_len ==  expected), "*** Expected: {}, got: {}".format(expected, res_len)
    
    # res_pars, res_sdev = myFitter.generate_parameter_list(1,1)
    # myFitter.lnlike(res_pars)
    
    if not debugging:
        result = myTestFitter.fit_groups(nfixed, 1) # result could be filename of samples

    print("Testing permute")
    
    # groups without amplitude
    nfree = 4
    nfixed = 2
    best_groups = np.array([
                  [0,0,0,0,0,0, 1,1,1,1, 0,0,0, 10],
                  [10,0,0,0,0,0, 1,1,1,1, 0,0,0, 10],
                  [10,10,10,10,10,10, 1,1,1,1, 0.5,0.5,0.5, 15],
                  [20,10,10,10,10,10, 1,1,1,1, -0.5,0.5,0.5, 15]
                ])

    # generate 5 amplitudes between 0 and 0.2
    free_amps = np.random.rand(4)/5
    fixed_amps = np.random.rand(1)/5
    best_sample = np.append(np.append(best_groups, free_amps), fixed_amps)
    ps = [p for p in multiset_permutations(range(nfree))]

    # for each permutation, confirm that permute() retrieves the best_sample
    for p in ps:
        permuted_sample = np.append(
                            np.append(best_groups[p], free_amps[p]),
                            fixed_amps)
        res = myTestFitter.permute(permuted_sample, best_sample, nfree, nfixed)
        assert(np.array_equal(res,best_sample))

