#! /usr/bin/env python
from chronostar.analyser import *
import numpy as np

# Test read_sampler
samples_file = 'logs/gf_bp_2_3_10_10.pkl'
# Check read_sampler returnss some variables
assert(read_sampler(samples_file) != 0), "Read_sampler not functioning"

chain, lnprob, pos, nfree, nfixed, nsteps, npars  = read_sampler(samples_file)
assert(np.size(chain) != 0), "Chain read in as zero"

assert(nfree == 3 and nfixed == 2)
