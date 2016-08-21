"""Test the fit_group program with a saved pickle from traceback

To use MPI, try:

mpirun -np 4 python test_fit_group.py
"""

import chronostar
from emcee.utils import MPIPool
import numpy as np
import sys 
import matplotlib.pyplot as plt 

star_params = chronostar.fit_group.read_stars("results/traceback_save.pkl")
    
if(True):
    using_mpi = True
    try:
        # Initialize the MPI-based pool used for parallelization.
        pool = MPIPool()
    except:
        print("MPI doesn't seem to be installed... maybe install it?")
        using_mpi = False
        pool=None
    
    if using_mpi:
        if not pool.is_master():
            # Wait for instructions from the master process.
            pool.wait()
            sys.exit(0)
        else:
            print("MPI available! - call this with e.g. mpirun -np 4 python fit_group.py")
   
    print("Up to fit_group") 
    beta_pic_group = np.array([-6.574, 66.560, 23.436, -1.327,-11.427, -6.527,\
        10.045, 10.319, 12.334,  0.762,  0.932,  0.735,  0.846, 20.589])
    plei_group = np.array([116.0,27.6, -27.6, 4.7, -23.1, -13.2, 20, 20, 20,\
                        3, 0, 0, 0, 70])

    dummy = chronostar.fit_group.lnprob_one_group(beta_pic_group, star_params, use_swig=True)
#    dummy = lnprob_one_group(plei_group, star_params, background_density=1e-10, use_swig=False)
        
    fitted_params = chronostar.fit_group.fit_one_group(star_params, pool=pool, use_swig=True)
    
    if using_mpi:
        # Close the processes.
        pool.close()
