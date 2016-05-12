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
    
    #Error checking (not needed in C once shown to work?) This shouldn't ever happen, as 
    #the determinants of the sum of positive definite matrices is
    #greater than the sum of their determinants    
    if (ApB_det < 0) | (B_det<0):
        pdb.set_trace()
        return -np.inf
    
    #Solve for c
    c = np.linalg.solve(ApB, AapBb)
    
    #Compute the overlap formula.

    overlap = np.exp(-0.5*(np.dot(b-c,np.dot(B,b-c)) + \
                           np.dot(a-c,np.dot(A,a-c)) )) 

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
