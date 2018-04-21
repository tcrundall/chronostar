"""
REDUNDANT
"""
#!/usr/bin/env python
#stars, times, xyzuvw, xyzuvw_cov = pickle.load(open('results/bp_TGAS2_traceback_save.pkl'))

from __future__ import print_function, division

import numpy as np
import chronostar.retired.fit_group as fit_group
import pickle                             # for reading in .pkl files
import corner                             # for producing the corner plots :O
import argparse                           # for calling script with arguments
import matplotlib.pyplot as plt           # for plotting the lnprob
import sys
from emcee.utils import MPIPool
"""
    the main testing bed of fit_group, utilising the beta pic moving group
    
    TO DO:
        - instead of printing median parameters with errors, save to file
"""

#Parsing arguments
parser = argparse.ArgumentParser()

parser.add_argument('-p', '--steps',  dest = 'p', default=10000,
                                    help='[5000] number of sampling steps')
parser.add_argument('-b', '--burnin', dest = 'b', default=2000,
                                    help='[1000] number of burn-in steps')
parser.add_argument('-d', '--bgdens', dest = 'd', default=2e-08,
                                    help='[2e-08] background density')

# from experience, the betapic data set needs ~1800 burnin steps to settle
# but it settles very well from then on
args = parser.parse_args()
nsteps = int(args.p)
burnin = int(args.b)
bgdens = float(args.d)

filestem = "bp_"+str(nsteps)+"_"+str(burnin)+"_"+str(bgdens)

def lnprob_plots(sampler):
    plt.plot(sampler.lnprobability.T)
    plt.title("lnprob of walkers")
    plt.savefig("plots/lnprob_"+filestem+".png")
    plt.clf()

def corner_plots(samples, best):
    fig = corner.corner(samples, labels=["X", "Y", "Z", "U", "V", "W",
                                         "dX", "dY", "dZ", "dVel",
                                         "xCorr", "yCorr", "zCorr", "age"],
                        truths=best)
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

def print_membership(stars, overlaps):
    # simply print the overlap with the group scaled such that ol + bgdens = 1
    for i in range(np.size(stars)):
        likeh = 100.0 * overlaps[i] / (overlaps[i] + bgdens)
        print("{}: {}%".format(stars[i], likeh))
        #pdb.set_trace()
    return 0

def write_results(samples):
    with open("logs/"+filestem+".log", 'w') as f:
        f.write("Log of output from bp with {} burn-in steps, {} sampling steps,\n"\
                    .format(burnin, nsteps) )
        f.write("\tand {} set for background dens\n".format(bgdens))
        f.write("\n")
        
        labels = ["X", "Y", "Z", "U", "V", "W",
                 "dX", "dY", "dZ", "dVel",
                 "xCorr", "yCorr", "zCorr", "age"]
        bf = calc_best_fit(samples)
        f.write(" _______ BETA PIC MOVING GROUP ________ \n")
        for i in range(14):
            f.write("{:5}: {:> 7.2f}  +{:>5.2f}  -{:>5.2f}\n".format(labels[i],
                                                    bf[i][0], bf[i][1], bf[i][2]) )

        ol_dynamic = fit_group.lnprob_one_group(fitted_group, star_params,
                                                use_swig=True, return_overlaps=True)
        
        bpstars = star_params["stars"]["Name1"][np.where(ol_dynamic > 1e-10)]
        allstars = star_params["stars"]["Name1"]
        ol_bp = ol_dynamic[np.where(ol_dynamic > 1e-10)]
        f.write("{} stars with overlaps > 1e-10:\n".format(np.size(bpstars)))
        f.write(str(bpstars)+"\n")

        f.write("\n")
        #print_membership(allstars, ol_dynamic)
        print("Just BP stars")
        print_membership(bpstars, ol_bp)

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
        pass
        # print("MPI available! - call this with e.g. mpirun -np 4 python fit_group.py")


stars, times, xyzuvw, xyzuvw_cov = \
        pickle.load(open('results/bp_TGAS2_traceback_save.pkl'))
star_params = fit_group.read_stars('results/bp_TGAS2_traceback_save.pkl')

beta_pic_group = np.array([-6.574, 66.560, 23.436, -1.327, \
                           -11.427,     0, 10.045, 10.319, \
                            12.334,     5,  0.932,  0.735, \
                            0.846, 20.589])

#xyzuvw (6), then xyz standard deviations (3), uvw_symmetrical_std (1), xyz_correlations (3)
nullish_init = np.array([ 0.0, 0.0,  0.0, -1.0, \
                        -10.0, 0.0, 10.0, 10.0, \
                         10.0, 0.5,  0.5,  0.5, \
                          0.5, 1.0])

null_init = np.array([0.0, 0.0, 0.0, 0.0, \
                      0.0,   0, 1.0, 1.0, \
                      1.0,   2, 1.0, 1.0, \
                      1.0, 1.0])

ol_swig = fit_group.lnprob_one_group(beta_pic_group, star_params, use_swig=True, return_overlaps=True)

sampler = fit_group.fit_one_group(star_params, init_mod=beta_pic_group, \
                                  nwalkers=32, nchain=nsteps, nburn=burnin, return_sampler=True, pool=pool, \
                                  init_sdev = np.array([1,1,1,1,1,1,1,1,1,.01,.01,.01,.1,1]), \
                                  background_density=bgdens, use_swig=True, plotit=False)

best_ix = np.argmax(sampler.flatlnprobability)
fitted_group = sampler.flatchain[best_ix]

#lnprob_plots(sampler)
#corner_plots(sampler.flatchain, fitted_group)
write_results(sampler.flatchain)

#print(ol_swig)
#print()
#print(fitted_group)

if using_mpi:
    #Close the processes
    pool.close()
