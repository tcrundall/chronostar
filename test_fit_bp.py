#!/usr/bin/env python
#stars, times, xyzuvw, xyzuvw_cov = pickle.load(open('results/bp_TGAS2_traceback_save.pkl'))

from __future__ import print_function, division

import numpy as np
import chronostar.fit_group as fit_group
import astropy.io.fits as pyfits            # for reading in .fts files
import pickle                               # for reading in .pkl files
import pdb


stars, times, xyzuvw, xyzuvw_cov = \
        pickle.load(open('results/bp_TGAS2_traceback_save.pkl'))
star_params = fit_group.read_stars('results/bp_TGAS2_traceback_save.pkl')

beta_pic_group = np.array([-6.574, 66.560, 23.436, -1.327,-11.427, 0,\
     10.045, 10.319, 12.334,  5,  0.932,  0.735,  0.846, 20.589])

ol_swig = fit_group.lnprob_one_group(beta_pic_group, star_params, use_swig=True, return_overlaps=True)

fitted_group = fit_group.fit_one_group(star_params, init_mod=beta_pic_group,\
        nwalkers=30,nchain=10000,nburn=1000, return_sampler=False,pool=None,\
        init_sdev = np.array([1,1,1,1,1,1,1,1,1,.01,.01,.01,.1,1]),\
        background_density=2e-12, use_swig=True, plotit=False)

ol_dynamic = fit_group.lnprob_one_group(fitted_group, star_params, use_swig=True, return_overlaps=True)

print("Stars with overlaps > 1e-10:")

print(star_params["stars"]["Name1"][np.where(ol_dynamic > 1e-10)])

pdb.set_trace()

print(ol_swig)
print()
print(fitted_group)
