#! /usr/bin/env python
"""Use traced back data to find best fitting groups.
"""
import chronostar.analyser as anl
from chronostar import groupfitter
import numpy as np
import time
import pickle
import pdb
import argparse
import sys
from emcee.utils import MPIPool
import matplotlib.pyplot as plt

def fit_groups(
        burnin, nsteps, nfree, nfixed, infile=None, debug=False, local=True, 
        noplots=False, tstamp=None, info=""
    ):
    # ---------------------------------
    # --   SETTING EVERYTHING UP     --
    # ---------------------------------
    
    # Setting up mpi
    using_mpi = True
    try:
        # Initialize the MPI-based pool used for parallelization.
        pool = MPIPool()
    except:
        print("MPI doesn't seem to be installed... maybe install it?")
        print("MPI available! - call this with e.g. mpirun -np 4 python"+
              " fit_groups.py")
        using_mpi = False
        pool=None
    
    if using_mpi:
        if not pool.is_master():
            # Wait for instructions from the master process.
            pool.wait()
            sys.exit(0)
        else:
            print("MPI available! - call this with e.g. mpirun -np 4 python"+
                  " fit_grups.py")
    
    if not local:
        data_dir    = '/short/kc5/data/'
        results_dir = '/short/kc5/results/' 
    else: 
        data_dir    = 'data/' 
        results_dir = 'results/' 
        #save_dir = '' 
     
    try:
        open(data_dir + "temp.txt",'w')
    except:
        print("*** If you're not running this on Raijin, with project kc5, ***\n"+
              "*** call with '-l' or '--local' flag                        ***")
        sys.exit()
        # raise UserWarning
    
    # a rough timestamp to help keep logs in order
    # if fixed_ages:
    #    tstamp = str(fixed_age).replace('.','_')\
    #               + "_" + str(int(time.time())%1000000)
    # else:
    if tstamp is None:
        tstamp = str(int(time.time())%1000000)
    print("Time stamp is: {}".format(tstamp))
    
    # --------------------------------
    # --         RUNNING FIT        --
    # --------------------------------
    
    bg = False
    samples, pos, lnprob =\
        groupfitter.fit_groups(
            burnin, nsteps, nfree, nfixed, infile=infile,
            bg=bg, loc_debug=debug)
    print("Run finished")
    
    # --------------------------------
    # --      ANALYSING RESULT      --
    # --------------------------------
    
    # get the number of parameters used from shape of samples array
    # samples.shape -> (nwalkers, nsteps, npars)
    nwalkers = np.shape(samples)[0]
    npars = np.shape(samples)[2]
    
    # reshape samples array to combine each walker's path into one chain
    flat_samples = np.reshape(samples, (nwalkers*nsteps, npars))
    
    # calculate the volume of each error ellipse
    vols =  np.zeros(flat_samples.shape[0])
    for i in range(vols.shape[0]):
        vols[i] = groupfitter.calc_average_eig(flat_samples[i])
    best_vol = anl.calc_best_fit(vols.reshape(-1,1))
    
    # converts the sample pars into physical pars, e.g. the inverse of stdev
    # is explored, the weight of the final group is implicitly derived
    cv_samples = anl.convert_samples(flat_samples, nfree, nfixed, npars)
    best_sample = cv_samples[np.argmax(lnprob)]
    
    # Calculate the median, positive error and negative error values
    best_fits = anl.calc_best_fit(cv_samples)
    
    # Generate log written into local logs/ directory
    anl.write_results(nsteps, nfree, nfixed, best_fits, tstamp, 
                      bw=best_vol, infile=infile, info=info)
    print("Logs written")
    
    # --------------------------------
    # --      PLOTTING RESULT       --
    # --------------------------------

    if not noplots:
    
        # In order to keep things generic w.r.t. raijin, the plotting data
        # is stored and a separate script is provided to come back and plot 
        # later
        file_stem = "{}_{}_{}".format(nfree, nfixed, nsteps)
        lnprob_pars = (lnprob, nfree, nfixed, tstamp)
        pickle.dump(
            lnprob_pars,
            open(results_dir+"{}_lnprob_".format(tstamp)+file_stem+".pkl",'w'))
        
        # Only include weights in corner plot if there are more than 1 group
        # being fitted. corner complains when asked to plot a fixed parameter
        # and 1 group will have a fixed weight of 1.
        weights=(nfixed+nfree > 1)
        cv_samples = anl.convert_samples(flat_samples, nfree, nfixed, npars)
        corner_plot_pars = (
            nfree, nfixed, cv_samples, lnprob,
            True, True, False, (not bg), weights, tstamp)
        pickle.dump(
            corner_plot_pars,
            open(results_dir+"{}_corner_".format(tstamp)+file_stem+".pkl",'w') )
        
        print("Plot data saved. Go back and plot with:\n"+
              "./plot_it.py -t {} -f {} -r {} -p {} -l\n".\
                  format(tstamp, nfree, nfixed, nsteps)+
              "   if not on raijin, or:\n"+
              "./plot_it.py -t {} -f {} -r {} -p {}\n".\
              format(tstamp, nfree, nfixed, nsteps))
    
    if using_mpi:
        pool.close()

    return best_sample, best_fits
