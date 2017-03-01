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
args = parser.parse_args() 
burnin = int(args.b) 
steps = int(args.p) 

# testing covariance matricies and their determinants


fixed_bg_group= [-15.41, -17.22, -21.32, -4.27, -14.39, -5.83,
                  73.34, 51.61, 48.83,
                  7.20,
                 -0.21, -0.09, 0.12,
                  0.0]
nfixed = 1

myFitter = GroupFitter(burnin=burnin, steps=steps, nfixed=nfixed, nfree=1,
                        fixed_groups=nfixed*[fixed_bg_group])

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

result = myFitter.fit_groups(nfixed, 1) # result could be filename of samples

#pdb.set_trace()
