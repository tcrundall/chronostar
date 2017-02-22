#!/usr/bin/env python

from chronostar.groupfitter import MVGaussian
from chronostar.groupfitter import GroupFitter
from chronostar.groupfitter import Star
from chronostar.groupfitter import Group
import pdb
import numpy as np
import matplotlib.pyplot as plt

params = [-6.574, 66.560, 23.436, -1.327,-11.427, -6.527,\
        10.045, 10.319, 12.334,  0.762,  0.932,  0.735,  0.846]
age = 20.589

print("____MVGAUSSIAN____")
myGauss = MVGaussian(params) 
print(myGauss)
print("MyGauss has params: {}".format(myGauss.params))

print("____STAR_____")
myStar = Star(params)
print(myStar)
print("MyStar has params: {}".format(myStar.params))

print("____GROUP____")
myGroup = Group(params, 0.3, age)
print(myGroup)
print("MyGroup has params: {}, amplitude: {} and age: {}".format(
                                    myGroup.params, myGroup.amplitude, myGroup.age))

myFitter = GroupFitter()

dummy_params = [-15.41, -17.22, -21.32, -4.27, -14.39, -5.83,
                              73.34, 51.61, 48.83,
                              7.20,
                             -0.21, -0.09, 0.12]

print(myFitter.lnprob(dummy_params))
result = myFitter.fitGroups()

plt.plot(myFitter.sampler.lnprobability.T)

pdb.set_trace()
