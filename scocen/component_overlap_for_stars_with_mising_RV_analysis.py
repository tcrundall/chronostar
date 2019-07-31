"""
Author: Marusa Zerjal, 2019 - 07 - 15

Take Sco-Cen components fitted to 6D data and make overlaps
(using covariance matrix) with stars missing radial velocities
in order to find more Sco-Cen candidates.

MZ: It fails in python2 (cannot import emcee).

"""

import numpy as np
import sys
sys.path.insert(0, '..')
from chronostar.component import SphereComponent
from chronostar import tabletool
from chronostar import expectmax
from astropy.table import Table, vstack, join



c = np.load('all_nonbg_scocen_comps.npy') # including LCC
print('components', c.shape)
print('Are there duplicate components?')

d = Table.read('data_table_cartesian_with_bg_ols_and_component_overlaps.fits')



for i in range(1, 16+1):
    mask = d['comp_overlap_%d'%i]>0.5
    print i, len(d[mask])



def compare_membership_probabilities_of_stars_with_known_radial_velocities():


    memb_usco = np.load('usco_res/final_membership.npy')
    data_usco = Table.read('usco_res/usco_run_subset.fit')
    for i in range(memb_usco.shape[1]-1):
        data_usco['Comp_USco_%d'%(i+1)] = memb_usco[:,i]
    data_usco['Comp_bg'] = memb_usco[:,-1]

    memb_ucl = np.load('ucl_res/final_membership.npy')
    data_ucl = Table.read('ucl_res/ucl_run_subset.fit')
    for i in range(memb_ucl.shape[1]-1):
        data_ucl['Comp_UCL_%d'%(i+1)] = memb_ucl[:,i]
    data_ucl['Comp_bg'] = memb_ucl[:,-1]

    data_memb = vstack([data_usco, data_ucl])
    print(data_memb)

    d = join(data_table, data_memb, keys='source_id')
    print(d)

