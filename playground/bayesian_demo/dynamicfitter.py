#! /usr/bin/env python
import numpy as np
import pdb
import pickle
import scipy.optimize as opt
import matplotlib.pyplot as plt
import data_generator as dg
#from fun import *

## Example of how to use optimiser to minimise for m
#def orig_func(x):
#    return 2*x
#
#xs = np.linspace(0,10,101)
#m0 = 2
#def fitting_func(m, xs):
#    return np.abs(np.sum(m*xs) - np.sum(orig_func(xs)))
#
#res = opt.minimize(fitting_func, m0, (xs))
#ms = np.linspace(-1,5,50)
#
#trace_back, n_time_steps, nstars, times, orig =\
#    pickle.load(open("data.pkl", 'r'))

def gaussian_fitter(pars, nstars, trace_back):
    npoints = 1000
    mu, sig = pars
    try:
        assert sig > 0
    except:
        pdb.set_trace()
    xs = np.linspace(-1000,1000,npoints)
    
    summed_stars = dg.group_pdf(xs, trace_back)
    gaussian_fit = nstars * dg.gaussian(xs, mu, sig)

    squared_diff = (summed_stars - gaussian_fit)**2
    return np.sum(squared_diff)

def lnprior(sig):
    """
    Generates a prior based on the stdev of fitted gaussian
    smaller sigmas are preferred since we want the group of stars
    to be dense. Extremely small sigmas should be penalised though
    """
    # @Mike
    min_sig = 2.0
    return 100 * np.log(sig / (min_sig**2 + sig**2))

def single_overlap(group_pars, star_pars):
    # @Mike
    mu_g, sig_g = group_pars
    mu_s, sig_s = star_pars
    numer = np.exp(-(mu_s - mu_g)**2 / (2*(sig_s**2 + sig_g**2)))
    denom = np.sqrt(2*np.pi*(sig_s**2 + sig_g**2))
    return numer/denom

def overlap(pars, nstars, trace_back):
    # @Mike
    mu, sig = pars
    
    total_overlap = 0
    for i in range(nstars):
        star_pars  = trace_back[i][0:2]
        total_overlap += np.log(single_overlap(pars, star_pars))
    return - total_overlap 
