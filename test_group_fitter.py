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

fixed_bg_group= [-15.41, -17.22, -21.32, -4.27, -14.39, -5.83,
                  73.34, 51.61, 48.83,
                  7.20,
                 -0.21, -0.09, 0.12]
nfixed = 1

myFitter = GroupFitter(burnin=burnin, steps=steps, nfixed=nfixed, nfree=1,
                        fixed_groups=nfixed*[fixed_bg_group])

#result = myFitter.fit_groups(nfixed, 1) # result could be filename of samples

#myAnalyser = SamplerAnalyser(result) # I could initialise an Analyser object
                                      # to investigate the sample/produce plots
                                      # etc. It would be useful to have them
                                      # separate so I can run a bunch of runs
                                      # automated and investigate afterwards

#myAnalyser.makePlots(show=True)
#myAnalyser.write

free_group_npars = 14

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
        expected = 14 * nfree + nfixed - 1
        assert(res_len ==  expected), "*** Expected: {}, got: {}".format(expected, res_len)

res_pars, res_sdev = myFitter.generate_parameter_list(1,1)
myFitter.lnlike(res_pars)

#pdb.set_trace()
