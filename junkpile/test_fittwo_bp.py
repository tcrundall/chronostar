#!/usr/bin/env python
#stars, times, xyzuvw, xyzuvw_cov = pickle.load(open('results/bp_TGAS2_traceback_save.pkl'))

from __future__ import print_function, division

import numpy as np
import chronostar.fit_group as fit_group
import astropy.io.fits as pyfits          # for reading in .fts files
import pickle                             # for reading in .pkl files
import pdb
try:
    using_corner = True
    import corner                             # for producing the corner plots :O
except:
    print("no corner plotting on Raijin.... :( ")
    using_corner = False

import argparse                           # for calling script with arguments
import matplotlib.pyplot as plt           # for plotting the lnprob

from emcee.utils import MPIPool
import sys

"""
    the main testing bed of fit_group, utilising the beta pic moving group
    
    TO DO:
        - add third group
        - use unbiased data set (i.e. not stars selected for being around BPMG
"""

#Parsing arguments
parser = argparse.ArgumentParser()

parser.add_argument('-p', '--steps',  dest = 'p', default=10000,
                                    help='[1000] number of sampling steps')
parser.add_argument('-b', '--burnin', dest = 'b', default=2000,
                                    help='[700] number of burn-in steps')
#parser.add_argument('-t', '--time', dest = 't',
#                                    help='[] specified time to fit to')
#parser.add_argument('-d', '--bgdens', dest = 'd', default=2e-08,
#                                    help='[2e-08] background density')


# from experience, the betapic data set needs ~1800 burnin steps to settle
# but it settles very well from then on (if starting with help)
args = parser.parse_args()
nsteps = int(args.p)
burnin = int(args.b)
#if args.t:
#    time = float(args.t)
#    istime = True
#else:
#    istime = False
#pdb.set_trace()
bgdens = False

filestem = "bp_two_"+str(nsteps)+"_"+str(burnin)

def lnprob_plots(sampler):
    plt.plot(sampler.lnprobability.T)
    plt.title("lnprob of walkers")
    plt.savefig("plots/lnprob_"+filestem+".png")
    plt.clf()

def corner_plots(samples):
    #fig = corner.corner(samples)
    #fig = corner.corner(samples, labels=["X", "Y", "Z", "U", "V", "W",
    #                                     "dX", "dY", "dZ", "dVel",
    #                                     "xCorr", "yCorr", "zCorr", 
    #                                     "X2", "Y2", "Z2", "U2", "V2", "W2",
    #                                     "dX2", "dY2", "dZ2", "dVel2",
    #                                     "xCorr2", "yCorr2", "zCorr2", "weight", "age"],
    #                     truths=best)
    fig = corner.corner(samples, labels=["X", "Y", "Z",
                                         "dX", "dY", "dZ",
                                         "weight", "age"])
    fig.savefig("plots/corner_"+filestem+".png")

def calc_best_fit(samples):
    return np.array( map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                     zip(*np.percentile(samples, [16,50,84], axis=0))) )

def print_results(samples):
    labels = ["X", "Y", "Z", "U", "V", "W",
             "dX", "dY", "dZ", "dVel",
             "xCorr", "yCorr", "zCorr", "age"]
    bf = calc_best_fit(samples)
    print(" _______ BETA PIC MOVING GROUP ________ ")
    for i in range(14):
        print("{:5}: {:> 7.2f}  +{:>5.2f}  -{:>5.2f}".format(labels[i],
                                                bf[i][0], bf[i][1], bf[i][2]) )

def calculate_membership(stars, ass_overlaps, bg_overlaps):
    likeh = 100.0 * ass_overlaps / (ass_overlaps+bg_overlaps)
    #pdb.set_trace()
    print("Stars with membership likelihood greater than 80%: {}"\
                        .format(np.size(np.where(likeh>80.0))))
    print("Stars with membership likelihood greater than 50%: {}"\
                        .format(np.size(np.where(likeh>50.0))))
    print("  out of {} stars".format(np.size(stars)))
    return 0

def print_membership(stars, overlaps):
    # simply print the overlap with the group scaled such that ol + bgdens = 1
    for i in range(np.size(stars)):
        likeh = 100.0 * overlaps[i] / (overlaps[i] + bgdens)
        print("{}: {}%".format(stars[i], likeh))
        #pdb.set_trace()
    return 0

def write_results(samples, stars, ass_overlaps, bg_overlaps):
    with open("logs/"+filestem+".log", 'w') as f:
        f.write("Log of output from bp with {} burn-in steps, {} sampling steps,\n"\
                    .format(burnin, nsteps) )
        f.write("\tand {} set for background dens\n".format(bgdens))
        f.write("Using starting parameters:\n{}".format(str(beta_pic_group)))
        f.write("\n")
        
        labels = ["X", "Y", "Z", "U", "V", "W",
                 "dX", "dY", "dZ", "dVel",
                 "xCorr", "yCorr", "zCorr",
                 "X2", "Y2", "Z2", "U2", "V2", "W2",
                 "dX2", "dY2", "dZ2", "dVel2",
                 "xCorr2", "yCorr2", "zCorr2", "weight", "age"] 
        bf = calc_best_fit(samples)
        f.write(" _______ BETA PIC MOVING GROUP ________ {starting parameters}\n")
        for i in range(len(labels)):
            f.write("{:6}: {:> 7.2f}  +{:>5.2f}  -{:>5.2f}\t\t\t{:>7.2f}\n".format(labels[i],
                                                    bf[i][0], bf[i][1], bf[i][2],
                                                    big_beta_group[i]) )
        likeh = 100.0 * ass_overlaps / (ass_overlaps+bg_overlaps)
        #pdb.set_trace()

        nstars = np.size(stars)
        defbp = np.size(np.where(likeh>80.0))
        maybp = np.size(np.where(likeh>50.0))

        f.write("Stars with membership likelihood greater than 80%: {} or {:5.2f}%\n"\
                            .format(defbp, 100.0 * defbp / nstars))
        f.write("Stars with membership likelihood greater than 50%: {} or {:5.2f}%\n"\
                            .format(maybp, 100.0 * maybp / nstars))
        f.write("  out of {} stars\n".format(nstars))

        #ol_dynamic = fit_group.lnprob_one_group(fitted_group, star_params,
        #                                        use_swig=True, return_overlaps=True)
        
        #bpstars = star_params["stars"]["Name1"][np.where(ol_dynamic > 1e-10)]
        #allstars = star_params["stars"]["Name1"]
        #ol_bp = ol_dynamic[np.where(ol_dynamic > 1e-10)]
        #f.write("{} stars with overlaps > 1e-10:\n".format(np.size(bpstars)))
        #f.write(str(bpstars)+"\n")

        #f.write("\n")
        #print_membership(allstars, ol_dynamic)
        #print("Just BP stars")
        #print_membership(bpstars, ol_bp)


stars, times, xyzuvw, xyzuvw_cov = \
        pickle.load(open('results/bp_TGAS2_traceback_save.pkl'))
star_params = fit_group.read_stars('results/bp_TGAS2_traceback_save.pkl')

beta_pic_group = np.array([-6.0, 66.0, 23.0, \
                            -1.0, -11.0,   0.0, \
                             10.0, 10.0, 12.0, 5, \
                            0.9,  0.7, 0.8, \
                            -35.0, 1.0, -30.0, -4.0, -15.0, -5.0, \
                            80.0, 60.0, 50.0, \
                            7, \
                            -0.2, 0.3, -0.1, \
                            0.30, \
                            23.0]) # birth time

# The fit being fitted by two gaussians
big_beta_group = np.array([-22, 34, 26, \
                             0.61, -14, 0.01, \
                            27, 35, 20, 3.6,\
                            0.39, 0.19, 0.18, \
                            -19, -22, -46, \
                            -6.3, -16.5, -6.9, \
                            107, 60, 47, \
                            9.2, \
                            -0.27, 0.02, 0.18, \
                            0.5, \
                            10.6])

big_beta_group = np.array([ -24.74, 34.35, 25.33,
                              0.49, -14.01, 0.39,
                             28.46, 32.98, 21.15, 2.66,
                              0.33, 0.30, 0.27,
                             -8.79, -19.53, -41.58,
                             -6.03, -16.13, -6.90,
                             87.48, 59.63, 43.47,
                              8.55, -0.22, 0.05,
                              0.12,
                              0.51,
                             11.45] )


using_mpi = True
try:
    # Initialize the MPI-based pool used for parallelization.
    pool = MPIPool()
except:
    print("Either MPI doesn't seem to be installed or you aren't running with MPI... ")
    using_mpi = False
    pool=None

if using_mpi:
    if not pool.is_master():
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)
else:
    print("MPI available for this code! - call this with e.g. mpirun -np 16 python test_fittwo_bp.py")


if False:
    pdb.set_trace()
    beta_pic_group = beta_pic_group[:-1]
    sampler = fit_group.fit_two_groups(star_params, init_mod=beta_pic_group,\
        nwalkers=60,nchain=nsteps, nburn=burnin, return_sampler=True,pool=None,\
        init_sdev = np.array([1,1,1,1,1,1,1,1,1,.01,.01,.01,.1,1,1,1,1,1,1,1,1,1,0.1,0.01,0.01,0.01,0.1,1]),\
        use_swig=True, plotit=False, t_ix=time)
else:
    #pdb.set_trace()
    sampler = fit_group.fit_two_groups(star_params, init_mod=big_beta_group,\
        nwalkers=60,nchain=nsteps, nburn=burnin, return_sampler=True,pool=None,\
        init_sdev = np.array([1,1,1,1,1,1,1,1,1,.01,.01,.01,.1,1,1,1,1,1,1,1,1,1,0.1,0.01,0.01,0.01,0.005,1]),\
        use_swig=True, plotit=False)

if using_mpi:
    # Close the processes
    pool.close()

best_ix = np.argmax(sampler.flatlnprobability)
fitted_group = sampler.flatchain[best_ix]

#extracting interesting parameters
chain = sampler.flatchain
xyzs = chain[:,0:3]
dxyzs = chain[:,6:9]
weight_and_age = chain[:,-2:]

chain_of_interest = np.hstack((np.hstack((xyzs, dxyzs)), weight_and_age)) 
lnprob_plots(sampler)
if using_corner:
    corner_plots(chain_of_interest)

overlaps_tuple = fit_group.lnprob_two_groups(fitted_group, star_params, return_overlaps=True)
all_stars = star_params["stars"]["Name1"]
calculate_membership(all_stars, overlaps_tuple[0], overlaps_tuple[1])

#age_T = np.reshape(age, (600,1))
#np.hstack((xyz, age_T))

#corner_plots(sampler.flatchain, fitted_group)
write_results(sampler.flatchain, all_stars, overlaps_tuple[0], overlaps_tuple[1])
pickle.dump((sampler.chain, sampler.lnprobability), open("logs/" + filestem + ".pkl", 'w'))
