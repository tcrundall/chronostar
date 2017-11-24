#! /usr/bin/env python
"""
Generates a varied set of synthetic astrometry datasets.
The initial group conditions are parametrised in the standard way.
The "measurement error" is currently a uniform 1% error for each measurement
(excluding RA and DEC obvy)
"""
from chronostar import synthesiser as syn
import numpy as np

group_pars = np.array([
    # X, Y, Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,N
    [ 0, 0, 0,0,0,0,10,10,10, 3,  0,  0,  0,20,30],
    [ 0,10, 0,0,0,0, 5, 5, 5, 3,0.6,0.5,0.0,20,40],
    [10, 0, 0,0,0,0,10,10,10, 3,0.4,0.2,0.0,20,50],
    [10,10, 0,0,0,0, 5, 5, 5, 3,  0,0.2,0.0,10,60],
    [10,10,10,0,0,0,10,10,10, 3,  0,0.2,0.0,10,20],
    [ 0,10, 0,0,0,2, 5, 5, 5, 3,0.6,0.5,0.0,20,41],
    [ 0,10, 0,0,0,0, 5, 5, 5, 3,0.6,0.5,0.0,40,42],
    [ 0,10,50,0,0,0, 5, 5, 5, 3,0.6,0.5,0.0,40,43],
    ])
error = 1.0
for i in range(len(group_pars)):
    syn.synthesise_data(1, group_pars[i], error)
    print("---- {} group(s) synthesised".format(i))

# synthesise a bunch of solo groups with varying error
# errors = [0.01, 0.02, 0.05, 0.1]

# for error in errors:

# # synthesise a bunch of solo groups
# for i in range(5):
#     syn.synthesise_data(1, group_pars[i], error)
# 
# # synthesise a bunch of paired groups
# for i in range(4):
#     syn.synthesise_data(2, group_pars[i:i+2], error)
# 
# # synthesise a couple of three groups
# syn.synthesise_data(3, group_pars[0:3], error)
# syn.synthesise_data(3, group_pars[2:5], error)

