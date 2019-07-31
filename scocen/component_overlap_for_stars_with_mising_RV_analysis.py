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
    maskRV = d['radial_velocity_error']<500.0
    mask = d['comp_overlap_%d'%i]>0.5

    print i, len(d[mask]), len(d[maskRV&mask])



def compare_membership_probabilities_of_stars_with_and_without_radial_velocities(d):
    """
    mask = d['comp_overlap_13'] > 0.5
    maskRV = d['radial_velocity_error'] < 500.0

    d9rv=d[maskRV&mask]
    d9norv=d[~maskRV&mask]

    print d9rv
    print d9norv

    import matplotlib.pyplot as plt

    fig=plt.figure()
    ax=fig.add_subplot(111)

    ax.scatter(d9rv['Z'], d9rv['W'], s=1, c='k')
    ax.scatter(d9norv['Z'], d9norv['W'], s=1, c='r')

    plt.savefig('compare.png')

    plt.show()
    """

    # Members in the overall data
    mask_members_d = d['comp_overlap_1']>1 # Everything is False
    for i in range(1, 16+1):
        mask_members_d = np.logical_or(mask_members_d, d['comp_overlap_%d' % i] > 0.5)


    memb_usco = np.load('usco_res/final_membership.npy')
    data_usco = Table.read('usco_res/usco_run_subset.fit')
    for i in range(memb_usco.shape[1]-1):
        data_usco['Comp_USco_%d'%(i+1)] = memb_usco[:,i]
    data_usco['Comp_bg'] = memb_usco[:,-1]


    #print 'USCO MEMBERS'
    #print data_usco[mask_members]

    memb_ucl = np.load('ucl_res/final_membership.npy')
    data_ucl = Table.read('ucl_res/ucl_run_subset.fit')
    for i in range(memb_ucl.shape[1]-1):
        data_ucl['Comp_UCL_%d'%(i+1)] = memb_ucl[:,i]
    data_ucl['Comp_bg'] = memb_ucl[:,-1]

    mask_members = data_ucl['Comp_UCL_1']>1 # Everything is False
    for i in range(1, 4+1):
        mask_members = np.logical_or(mask_members, data_usco['Comp_UCL_%d'%i]>0.5)

    data_memb = vstack([data_usco, data_ucl])
    #print(data_memb)

    dall = join(d, data_memb, keys='source_id')
    print(d.colnames)
    print(data_memb.colnames)
    print(dall.colnames)



    mask_members_usco = dall['Comp_USco_1']>1 # Everything is False
    for i in range(1, 4+1):
        mask_members_usco = np.logical_or(mask_members_usco, dall['Comp_USco_%d'%i]>0.5)

    mask_members_ucl = dall['Comp_UCL_1']>1 # Everything is False
    for i in range(1, 4+1):
        mask_members_ucl = np.logical_or(mask_members_ucl, dall['Comp_UCL_%d'%i]>0.5)

    mask_members = dall['Comp_UCL_1']>1 # Everything is False
    for i in range(1, 16+1):
        mask_members = np.logical_or(mask_members, dall['comp_overlap_%d'%i]>0.5)

    print len(dall[mask_members_usco]), len(dall[mask_members_ucl]), len(dall[mask_members])


compare_membership_probabilities_of_stars_with_and_without_radial_velocities(d)