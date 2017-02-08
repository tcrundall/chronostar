#!/usr/bin/env python
#stars, times, xyzuvw, xyzuvw_cov = pickle.load(open('results/bp_TGAS2_traceback_save.pkl'))

from __future__ import print_function, division

#import matplotlib
#matplotlib.use('Agg')		  # sposed to stop plt.savefig() from displaying
import numpy as np
import chronostar.fit_group as fit_group
import astropy.io.fits as pyfits  # for reading in .fts files
import pickle                     # for reading in .pkl files
import pdb
import argparse                   # for calling script with arguments
import corner                     # for producing the corner plots :O
import matplotlib.pyplot as plt   # for plotting the lnprob
import sys
from emcee.utils import MPIPool

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
ndims = 43
#if args.t:
#    time = float(args.t)
#    istime = True
#else:
#    istime = False
#pdb.set_trace()
bgdens = False

filestem = "partemp_bp_three_"+str(nsteps)+"_"+str(burnin)

def lnprob_plots(sampler):
    plt.plot(sampler.lnprobability.T)
    plt.title("lnprob of walkers")
    plt.savefig("plots/lnprob_"+filestem+".png")
    plt.clf()

def corner_plots(samples, init):
    wanted_pars = [6,7,8,9,13,14,21,22,23,24,28,29]
    labels=["dX1", "dY1", "dZ1", "dV1", "age1", "weight1", "dX2", "dY2", "dZ2", "dV2", "age2", "weight2"]
    fig = corner.corner(samples[:,wanted_pars], truths=init[wanted_pars], labels=labels)
    fig.savefig("plots/corner_"+filestem+".png")

def calc_best_fit(samples):
    return np.array( map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                     zip(*np.percentile(samples, [16,50,84], axis=0))) )


def print_membership(stars, overlaps):
    # simply print the overlap with the group scaled such that ol + bgdens = 1
    for i in range(np.size(stars)):
        likeh = 100.0 * overlaps[i] / (overlaps[i] + bgdens)
        print("{}: {}%".format(stars[i], likeh))
        #pdb.set_trace()
    return 0

def write_results(samples, stars, g1_overlaps, g2_overlaps, bg_overlaps):
    with open("logs/"+filestem+".log", 'w') as f:
        f.write("Log of output from bp with {} burn-in steps, {} sampling steps,\n"\
                    .format(burnin, nsteps) )
        f.write("\tand {} set for background dens\n".format(bgdens))
        f.write("Using starting parameters:\n{}".format(str(beta_pic_group)))
        f.write("\n")
        
        labels = ["X1", "Y1", "Z1", "U1", "V1", "W1",
                 "dX1", "dY1", "dZ1", "dVel1",
                 "xCorr1", "yCorr1", "zCorr1",
                 "age1", "weight1",
                 "X2", "Y2", "Z2", "U2", "V2", "W2",
                 "dX2", "dY2", "dZ2", "dVel2",
                 "xCorr2", "yCorr2", "zCorr2",
                 "age2", "weight2",
                 "BGX", "BGY", "BGZ", "BGU", "BGV", "BGW",
                 "BGdX", "BGdY", "BGdZ", "BGdVel",
                 "BGxCorr", "BGyCorr", "BGzCorr" ] 
        bf = calc_best_fit(samples)
        f.write(" _______ BETA PIC MOVING GROUP ________ {starting parameters}\n")
        for i in range(len(labels)):
            f.write("{:8}: {:> 7.2f}  +{:>5.2f}  -{:>5.2f}\t\t\t{:>7.2f}\n".format(labels[i],
                                                    bf[i][0], bf[i][1], bf[i][2],
                                                    big_beta_group[i]) )
        total_ols = g1_overlaps + g2_overlaps + bg_overlaps
        likeh1 = 100.0 * g1_overlaps / total_ols
        likeh2 = 100.0 * g2_overlaps / total_ols
        #pdb.set_trace()

        nstars = np.size(stars)
        defg1 = np.size(np.where(likeh1>80.0))
        mayg1 = np.size(np.where(likeh1>50.0))
        defg2 = np.size(np.where(likeh2>80.0))
        mayg2 = np.size(np.where(likeh2>50.0))

        f.write("Stars with group 1 membership likelihood greater than 80%: {} or {:5.2f}%\n"\
                            .format(defg1, 100.0 * defg1 / nstars))
        f.write("Stars with group 1 membership likelihood greater than 50%: {} or {:5.2f}%\n"\
                            .format(mayg1, 100.0 * mayg1 / nstars))
        f.write("Stars with group 2 membership likelihood greater than 80%: {} or {:5.2f}%\n"\
                            .format(defg2, 100.0 * defg2 / nstars))
        f.write("Stars with group 2 membership likelihood greater than 50%: {} or {:5.2f}%\n"\
                            .format(mayg2, 100.0 * mayg2 / nstars))
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

#fit from fit_two plus original beta pic fit
big_beta_group = np.array([-22, 34, 26, 0.61, -14, 0.01, \
                            27, 35, 20,\
                            3.6,\
                            0.39, 0.19, 0.18, \
                            10.6, 0.5, \
                            -6.0, 66.0, 23.0, -1.0, -11.0, 0.0, \
                            10.0, 10.0, 12.0,\
                            5, \
                            0.9,  0.7, 0.8, \
                            23.0, 0.15, \
                            -19, -22, -46, -6.3, -16.5, -6.9, \
                            107, 60, 47, \
                            9.2, \
                            -0.27, 0.02, 0.18])

#taking the fit from bp_three_6000_3000.log (172mins)
big_beta_group = np.array([ -25.17, 45.34, 13.39, 1.01, -15.37, 2.20,    \
                             17.93, 48.04, 17.82,                        \
                             16.79,                                      \
                              0.36, 0.11, 0.26,                          \
                             14.66, 0.23,                                \
                             -1.01, 59.69, 26.63, -0.41, -11.58, -0.13,  \
                             14.82, 20.90, 15.79,                        \
                              1.61,                                      \
                              0.58, 0.79, 0.56,                          \
                             19.13, 0.12,                                \
                            -15.31, -17.73, -26.06, -4.04, -15.44, -5.53,\
                             83.45, 56.69, 49.09,                        \
                              7.56,                                      \
                             -0.23, -0.09, 0.14])

#taking the top 1% best lnprob fits and taking the medians
big_beta_group - np.array([  -23.57, 40.88, 9.51, 1.16, -15.60, 1.92,\
                             27.45, 44.64, 28.66,\
                             31.48,\
                             0.27, 0.10, 0.55,\
                             14.56, 0.21,\
                             -3.25, 59.41, 27.43, -0.56, -11.26, -0.35,\
                             15.62, 19.05, 12.85,\
                             1.60,\
                             0.46, 0.99, 0.35,\
                             19.99, 0.13,\
                             -17.36, -16.99, -24.09, -3.63, -15.84, -5.03,\
                             77.36, 53.67, 52.56,\
                             7.24,\
                             -0.23, -0.06, 0.20])

#the fit with highest lnprob from bp_three_20000_10000.log (-10656.8)
big_beta_group = np.array([ -22.24, 69.32, 11.49, 0.89, -15.44, 2.38,
                            58.24, 90.23, 53.96,
                            47.62,
                            0.71, -0.44, 0.57,
                            17.82, 0.26,
                            -0.73, 72.37, 26.99, 0.47, -10.99, -0.14,
                            2.07, 20.41, 16.50,
                            1.32,
                            0.61, 0.93, 0.28,
                            20.37, 0.08,
                            -15.41, -17.22, -21.32, -4.27, -14.39, -5.83,
                            73.34, 51.61, 48.83,
                            7.20,
                            -0.21, -0.09, 0.12])

init_sdev = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.01, 0.01, 0.01, 1, 0.05, \
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.01, 0.01, 0.01, 1, 0.05, \
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.01, 0.01, 0.01])

print("About to set up mpi")

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
    print("MPI available for this code! - call this with e.g. mpirun -np 16 python test_fitthree_bp.py")

print("About to run sampler")

#pdb.set_trace()
sampler = fit_group.fit_three_groups(star_params, init_mod=big_beta_group,\
    nwalkers=100,nchain=nsteps, nburn=burnin, return_sampler=True,pool=pool,\
    init_sdev = init_sdev,
    use_swig=True, plotit=False)

if using_mpi:
    # Close the processes
    pool.close()

print("Closed MPI processes")

flatlnprob = np.reshape(sampler.lnprobability, (-1, nsteps))
best_ix = np.argmax(flatlnprob)
flatchain = np.reshape(sampler.chain, (-1, ndims))
fitted_group = flatchain[best_ix]

print("About to dump data")

#extracting interesting parameters
#chain = sampler.flatchain

#pdb.set_trace()
#lnprob_plots(sampler)

#overlaps_tuple = fit_group.lnprob_three_groups(fitted_group, star_params, return_overlaps=True)
#all_stars = star_params["stars"]["Name1"]

#age_T = np.reshape(age, (600,1))
#np.hstack((xyz, age_T))

pickle.dump((sampler.chain, sampler.lnprobability), open("logs/" + filestem + ".pkl", 'w'))
#corner_plots(flatchain, big_beta_group)
#write_results(sampler.flatchain, all_stars, overlaps_tuple[0], overlaps_tuple[1], overlaps_tuple[2])
