#! /usr/bin/env python
from chronostar.retired.analyser import *
import numpy as np

plotit = False
print("___ Testing chronostar/analyser.py ___")

# Test read_sampler
samples_file = 'logs/archive/gf_bp_2_3_10_10.pkl'

# Check read_sampler returnss some variables
print("Testing read_sampler()")
assert(read_sampler(samples_file) != 0), "Read_sampler not functioning"

chain, lnprob, pos, nfree, nfixed, nsteps, npars  = read_sampler(samples_file)
assert(np.size(chain) != 0), "Chain read in as zero"
nwalkers = np.shape(chain)[0]

flatchain = np.reshape(chain, (nwalkers*nsteps,npars))
flatlnprob = np.reshape(lnprob, (nwalkers*nsteps))

assert(nfree == 3 and nfixed == 2)

print("Testing group_metric()")
good_pars = np.array([
              [0,0,0,0,0,0, 1,1,1,1, 0,0,0, 10, 0.5],
              [10,0,0,0,0,0, 1,1,1,1, 0,0,0, 10, 0.5],
              [10,10,10,10,10,10, 1,1,1,1, 0.5,0.5,0.5, 15, 0.5],
              [20,10,10,10,10,10, 1,1,1,1, -0.5,0.5,0.5, 15, 0.5]
            ])

bad_pars = [ [0,0,0,0,0,0,-1,1,1,1,0,0,0,10,0.5], # negative dX
              [10,10,10,10,10,10,1,1,1,1,1.5,0.5,0.5, 15, 0.5],# xycorr > 1
              [0,0,0,0,0,0,1,1,1,1, 0,0,0,10,1.5], # amplitude > 1
              [10,10,10,10,10,10,1,1,1,1,0.5,0.5,0.5,-15,0.5]# negative age
             ]
assert(group_metric(good_pars[0], good_pars[0]) == 0.0),\
        group_metric(good_pars[0], good_pars[0])
assert(group_metric(good_pars[1], good_pars[1]) == 0.0),\
        group_metric(good_pars[1], good_pars[1])
ngood_pars = 4
results = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        results[i,j] = group_metric(good_pars[i], good_pars[j])

# checking triangle inequality
for i in range(4):
    for j in range(i,4):
        for k in range(4):
            if (k != j and k != i):
                assert (results[i,j] <= results[i,k] + results[k,j]),\
                        "i: {}, j: {}, k: {}\n".format(i,j,k) + \
                        "{} , {}, {}".format(results[i,j], results[i,k],
                                             results[k,j])

print("Testing permute()")
# groups without amplitude
nfree_test = 4
nfixed_test = 2
best_groups = np.array([
              [0,0,0,0,0,0, 1,1,1,1, 0,0,0, 10],
              [10,0,0,0,0,0, 1,1,1,1, 0,0,0, 10],
              [10,10,10,10,10,10, 1,1,1,1, 0.5,0.5,0.5, 15],
              [20,10,10,10,10,10, 1,1,1,1, -0.5,0.5,0.5, 15]
            ])

# generate 6 amplitudes between 0 and 0.2
# note that these samples won't satisfy all amplitudes summing to 1
free_amps = np.random.rand(nfree_test)/5
fixed_amps = np.random.rand(nfixed_test)/5
best_sample = np.append(np.append(best_groups, free_amps), fixed_amps)
ps = [p for p in multiset_permutations(range(nfree_test))]

# for each permutation, confirm that permute() retrieves the best_sample
for p in ps:
    permuted_sample = np.append(
                        np.append(best_groups[p], free_amps[p]),
                        fixed_amps)
    res = permute(permuted_sample, best_sample, nfree_test, nfixed_test)
    assert(np.array_equal(res,best_sample))

file_stem = "{}_{}_{}_{}".format(nfree, nfixed, nsteps, nwalkers)

realigned_samples, permute_count = realign_samples(flatchain, flatlnprob,
                                                   nfree, nfixed, npars)

print("Permutations: {} out of: {}".format(permute_count,
                                           np.shape(flatchain)[0]) )
assert(np.shape(realigned_samples)[1] == np.shape(flatchain)[1]),\
            "Difference: {}".format(np.shape(realigned_samples)[1] -\
                                    np.shape(flatchain)[1])

print("Testing convert_samples()")
converted_samples = convert_samples(realigned_samples, nfree, nfixed, npars)
# 6:10 is where the stdevs are, groups are spaced 14 parameters apart
for i in range(nfree):
    assert(np.allclose(  realigned_samples[:, 14*i+6 : 14*i+10],
                       1/converted_samples[:, 14*i+6 : 14*i+10]))

assert(np.shape(converted_samples)[1] == np.shape(flatchain)[1] + 1),\
            "Difference: {}".format(np.shape(realigned_samples)[1] -\
                                    np.shape(flatchain)[1])
assert(np.array_equal(np.sum(converted_samples[:,-(nfree+nfixed):],axis=1),
                      np.ones(nwalkers*nsteps)) )

print("Testing calc_best_fit()")
best_fit = calc_best_fit(converted_samples)
assert(best_fit.shape == (converted_samples.shape[1], 3))

for i in range(converted_samples.shape[1]):
    assert(best_fit[i,0] == np.median(converted_samples[:,i]))


tstamp = "test"
if plotit:
    print("Testing plot_lnprob")
    plot_file_stem = "plots/lnprob_"  + file_stem

    plot_lnprob(lnprob, nfree, nfixed, tstamp)

    print("Testing generate_param_mask")
    assert(np.size(generate_param_mask(nfree, nfixed,
                                       True, False, False, False, True))
           == converted_samples.shape[1])
    print("Testing plot_corner()")
    # Test plot_corner
    plot_corner(
        nfree, nfixed, converted_samples, lnprob, weights=True, tstamp=tstamp)

print("Testing save_results()")
save_results(nfree, nfixed, nsteps, nwalkers, converted_samples, tstamp)

print("Testing generate_param_mask()")
print("FAILED... need a unit test for generate_param_mask()")
#assert False, "Need a unit test for generate_param_mask()"

print("___ chronostar/analyser.py passing all tests ___")
