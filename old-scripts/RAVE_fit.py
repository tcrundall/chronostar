#!/usr/bin/env python

import sys
sys.path.insert(0, '..')

import chronostar.groupfitter as gf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--tbfile',  dest = 'f',
                    help='tb_file being fitted to')
args = parser.parse_args()

tbfile = args.f

with open(tbfile, 'r') as fp:
    stars, ts, xyzuvw, xyzuvw_cov = pickle.load(fp)

init_pars=None
init_pars = np.array([
    -2.77515851e+01,  -5.12709248e+01,  -3.51019638e+01,
     5.15272650e+00,   5.38943567e+00,   5.44265710e-01,
     3.29030694e-02,   1.78098988e-02,   1.51952143e-02,
     6.65467580e-02,   9.99999855e-01,   9.99999874e-01,
     9.99999992e-01,   0.00000000e+00
])


pdb.set_trace()
print("About to fit...")
best_fit, chain, _ = gf.fit_group(
    tbfile, plot_it=True, fixed_age=0.0, init_pars=init_pars
)

with open("temp_results/fit_result.pkl", 'w') as fp:
    pickle.dump((best_fit, chain), fp)

#pdb.set_trace()
