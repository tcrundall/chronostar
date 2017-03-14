#!/usr/bin/env python

from chronostar.groupfitter import MVGaussian
from chronostar.groupfitter import GroupFitter
from chronostar.groupfitter import Star
from chronostar.groupfitter import Group
import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt

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
args = parser.parse_args() 
burnin = int(args.b) 
steps = int(args.p) 
nfree = int(args.free)
nfixed = int(args.fixed)
debugging = args.d
test_run  = args.t

# pdb.set_trace()
# testing covariance matricies and their determinants

fixed_bg_group= [-15.41, -17.22, -21.32, -4.27, -14.39, -5.83,
                 1/73.34, 1/51.61, 1/48.83,
                 1/7.20,
                 -0.21, -0.09, 0.12,
                  0.0]

print("Initialising myFitter...")
if not test_run:
    myFitter = GroupFitter(burnin=burnin, steps=steps, nfixed=nfixed,
                           nfree=nfree, fixed_groups=nfixed*[fixed_bg_group])
    # result could be filename of samples
    result = myFitter.fit_groups(nfixed, nfree)

if (test_run):
    myTestFitter = GroupFitter(burnin=10, steps=10, nfixed=1, nfree=1,
                        fixed_groups = [fixed_bg_group])
    print("About to test lnprior")
    
    # Testing lnprior
    good_pars = [ [0,0,0,0,0,0, 1,1,1,1, 0,0,0, 10, 0.5],
                  [10,10,10,10,10,10, 1,1,1,1, 0.5,0.5,0.5, 15, 0.5]
                ]
    
    bad_pars = [ [0,0,0,0,0,0,-1,1,1,1,0,0,0,10,0.5], # negative dX
                  [10,10,10,10,10,10,1,1,1,1,1.5,0.5,0.5, 15, 0.5],# xycorr > 1
                  [0,0,0,0,0,0,1,1,1,1, 0,0,0,10,1.5], # amplitude > 1
                  [10,10,10,10,10,10,1,1,1,1,0.5,0.5,0.5,-15,0.5]# negative age
                 ]
    
    #pdb.set_trace()
    for i in range(2):
        assert(myFitter.lnprior(good_pars[i]) == 0)
    
    
    for pars in bad_pars:
        assert(myFitter.lnprior(pars) == -np.inf), "par: {}".format(pars)
    
    print("lnprior seems fine...")
    
    #failed_sample = [ -8.47140941e+00,  6.09690031e+00, -1.64964204e+01, -8.16606704e+00,
    #                  -7.45727052e+00, -9.74301271e+00,  3.27440988e+01,  4.21552194e+01,
    #                   3.52616945e+01, -2.80981542e+00,  2.26040337e-01,  2.35967711e-02,
    #                  -1.98462993e-01, -2.67278810e-02]
    #
    #failed_lnprob = myFitter.lnprob(failed_sample)
    
    #myAnalyser = SamplerAnalyser(result) # I could initialise an Analyser object
                                          # to investigate the sample/produce plots
                                          # etc. It would be useful to have them
                                          # separate so I can run a bunch of runs
                                          # automated and investigate afterwards
    
    #myAnalyser.makePlots(show=True)
    #myAnalyser.write
    
    free_group_npars = 15
    
    max_nfixed = nfixed
    print("Testing generate_parameter_list()")
    for i in range(1,3):
        for nfree in range(1,3):
            nfixed = i
            print("-- {} fixed and {} free".format(nfixed, nfree))
            res_pars, res_sdev = myFitter.generate_parameter_list(nfixed,nfree)
            res_len = len(res_pars)
            if nfixed > max_nfixed:
                nfixed = max_nfixed
            expected = free_group_npars * nfree + nfixed - 1
            assert(res_len ==  expected), "*** Expected: {}, got: {}".format(expected, res_len)
    
    # res_pars, res_sdev = myFitter.generate_parameter_list(1,1)
    # myFitter.lnlike(res_pars)
    
    if not debugging:
        result = myTestFitter.fit_groups(nfixed, 1) # result could be filename of samples

#pdb.set_trace()

# Failed param list:
# failed_params = [   6.26384977e+02,  -9.98266769e+02,  -1.31659708e+02,  1.36050771e+03,
#    1.02290243e+02, -6.75402094e+02,   4.68336488e+02,   6.31595410e+01,
#    3.27252726e+02,  1.14134135e+02,   5.35106689e-01,  -6.83504092e-01,
#    1.23966913e-01,  4.07786862e+01,   1.25097761e-03]
# 
# 
# 
#     return self.f(x, *self.args, **self.kwargs)
#   File "/home/tcrun/chronostar/chronostar/groupfitter.py", line 451, in lnprob
#     return lp + self.lnlike(pars)
#   File "/home/tcrun/chronostar/chronostar/groupfitter.py", line 419, in lnlike
#     self.interp_icov(model_groups[i].age)
#   File "/home/tcrun/chronostar/chronostar/groupfitter.py", line 554, in interp_icov
#     self.STAR_MNS[:,ix0+1]*frac
# IndexError: index 41 is out of bounds for axis 1 with size 41
