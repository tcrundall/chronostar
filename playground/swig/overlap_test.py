"""This program takes an initial model for a stellar association and uses an affine invariant
Monte-Carlo to fit for the group parameters.

A group fitter, called after tracing orbits back.

This group fitter will find the best fit 6D error ellipse and best fit time for
the group formation based on Bayesian analysis, which in this case involves
computing overlap integrals. 
    
TODO:
0) Once the group is found, output the probability of each star being in the group.
1) Add in multiple groups 
2) Change from a group to a cluster, which can evaporate e.g. exponentially.
3) Add in a fixed background which is the Galaxy (from Robin et al 2003).

To use MPI, try:

mpirun -np 2 python fit_group.py

Note that this *doesn't* work yet due to a "pickling" problem.
"""

from __future__ import print_function, division

import emcee
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
import overlap #&TC
import time    #&TC
import overlap
from emcee.utils import MPIPool
   
def compute_overlap(A,a,A_det,B,b,B_det):
    """Compute the overlap integral between a star and group mean + covariance matrix
    in six dimensions, including some temporary variables for speed and to match the 
    notes.
    
    This is the first function to be converted to a C program in order to speed up."""
    #Preliminaries - add matrices together. This might make code more readable? 
    #Or might not.
    ApB = A + B
    AapBb = np.dot(A,a) + np.dot(B,b)
    
    #Compute determinants.
    ApB_det = abs(np.linalg.det(ApB)) # temporarily doctoring determinants

    print("ApB_det is: " + str(ApB_det))
    
    #Error checking (not needed in C once shown to work?) This shouldn't ever happen, as 
    #the determinants of the sum of positive definite matrices is
    #greater than the sum of their determinants    
    if (ApB_det < 0) | (B_det<0):
        pdb.set_trace()
        return -np.inf
    
    #Solve for c
    c = np.linalg.solve(ApB, AapBb)
    print("C vector is:")
    print(c)
    
    #Compute the overlap formula.

    print("A (a-c) is:")
    print(np.dot(A, a-c))
    print("(a-c) A (a-c) is:")
    print(np.dot(a-c, (np.dot(A, a-c))))
    print("B (b-c) is:")
    print(np.dot(B, b-c))
    print("(b-c) B (b-c) is:")
    print(np.dot(b-c, (np.dot(B, b-c))))
    overlap = np.exp(-0.5*(np.dot(b-c,np.dot(B,b-c)) + \
                           np.dot(a-c,np.dot(A,a-c)) )) 
    print("Result before sqrt_factor is: " + str(overlap))


    sqrt_factor = np.sqrt(B_det*A_det/ApB_det)/(2*np.pi)**3.0
    print("Sqrt_factor is: " + str(sqrt_factor))

    overlap *= np.sqrt(B_det*A_det/ApB_det)/(2*np.pi)**3.0
    
    return overlap

def main():
    A = np.random.random( (6,6) )
    a = np.random.random(6)
    A_det = abs(np.linalg.det(A)) # temporarily doctoring determinants
    B = np.random.random( (6,6) )
    b = np.random.random(6)
    B_det = abs(np.linalg.det(B)) # temporarily doctoring determinants

    print("Python: _____________")
    print(compute_overlap(A, a, A_det, B, b, B_det))

    print("C Module: ___________")
    print(overlap.get_overlap(A.flatten().tolist(), a.flatten().tolist(), \
                                A_det,                                    \
                              B.flatten().tolist(), b.flatten().tolist(), \
                                B_det))

main()
