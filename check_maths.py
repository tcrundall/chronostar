import chronostar
from chronostar.fit_group import compute_overlap as co
import numpy as np
#from math import isclose
import pdb

def new_co(A_cov,a_mn,B_cov,b_mn):
    """Compute the overlap integral between a star and group mean + covariance matrix
    in six dimensions, including some temporary variables for speed and to match the 
    notes.
    
    This is the first function to be converted to a C program in order to speed up."""

    AcpBc = (A_cov + B_cov)
    AcpBc_det = np.linalg.det(AcpBc)

    AcpBc_i = np.linalg.inv(AcpBc)
    #amn_m_bmn = a_mn - b_mn

    overlap = np.exp(-0.5 * (np.dot(a_mn-b_mn, np.dot(AcpBc_i,a_mn-b_mn) )) )
    overlap *= 1.0/((2*np.pi)**3.0 * np.sqrt(AcpBc_det))
    return overlap

def misc()
    #Preliminaries - add matrices together. This might make code more readable? 
    #Or might not.
    ApB = A + B
    AapBb = np.dot(A,a) + np.dot(B,b)

    #Compute determinants.
    ApB_det = np.linalg.det(ApB)

    #Solve for c
    c = np.linalg.solve(ApB, AapBb)

    #Compute the overlap formula.
    overlap = np.exp(-0.5*(np.dot(b-c,np.dot(B,b-c)) + \
                           np.dot(a-c,np.dot(A,a-c)) ))

    overlap *= np.sqrt(B_det*A_det/ApB_det)/(2*np.pi)**3.0

    return overlap
 
star_params = chronostar.fit_group.read_stars("results/bp_TGAS2_traceback_save.pkl")

icov = star_params["xyzuvw_icov"]
cov = star_params["xyzuvw_cov"]
mean = star_params["xyzuvw"]
det = star_params["xyzuvw_icov_det"]

for i in range(0,407):
    A_cov = cov[i,0]
    A_icov = icov[i,0]
    a_mn = mean[i,0]
    A_idet = det[i,0]
    
    B_cov = cov[i,1]
    B_icov = icov[i,1]
    b_mn = mean[i,1]
    B_idet = det[i,1]

    mikes_ol = co(A_icov,a_mn,A_idet,B_icov,b_mn,B_idet)
    tims_ol = new_co(A_cov,a_mn,B_cov,b_mn)

    if ( (mikes_ol - tims_ol)/mikes_ol > 1e-10):
        print("Discrepancy!!!")
        print("Difference: {}".format((mikes_ol - tims_ol)/mikes_ol))

