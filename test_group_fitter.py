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

myFitter = GroupFitter(burnin=burnin, steps=steps)

dummy_params = [-15.41, -17.22, -21.32, -4.27, -14.39, -5.83,
                              73.34, 51.61, 48.83,
                              7.20,
                             -0.21, -0.09, 0.12]

print(myFitter.lnprob(dummy_params))

#pdb.set_trace()

result = myFitter.fit_groups()

#pdb.set_trace()
