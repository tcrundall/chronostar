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


maskRV = d['radial_velocity_error']<500.0
for i in range(1, 15+1):
    mask = d['comp_overlap_%d'%i]>0.5

    print i, len(d[mask]), len(d[maskRV&mask])
mask = d['comp_overlap_bg']>0.5
print 'bg', len(d[mask], len(d[maskRV&mask]))


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
        mask_members = np.logical_or(mask_members, data_ucl['Comp_UCL_%d'%i]>0.5)

    data_memb = vstack([data_usco, data_ucl])
    #print(data_memb)

    # MERGE TABLES
    dall = join(d, data_memb, keys='source_id')

    mask_members_usco = dall['Comp_USco_1']>1 # Everything is False
    for i in range(1, 4+1):
        mask_members_usco = np.logical_or(mask_members_usco, dall['Comp_USco_%d'%i]>0.5)

    mask_members_ucl = dall['Comp_UCL_1']>1 # Everything is False
    for i in range(1, 4+1):
        mask_members_ucl = np.logical_or(mask_members_ucl, dall['Comp_UCL_%d'%i]>0.5)

    mask_members = dall['Comp_UCL_1']>1 # Everything is False
    for i in range(1, 16+1):
        mask_members = np.logical_or(mask_members, dall['comp_overlap_%d'%i]>0.5)

    print len(dall[mask_members_usco]), len(dall[mask_members_ucl]), 'sum', len(dall[mask_members_usco])+len(dall[mask_members_ucl]), 'Tims members:', len(dall[mask_members])

    mask_Tim = np.logical_or(mask_members_ucl, mask_members_usco)
    mask_norv = mask_members

    # Test if stars are members in both analyses
    mask_both = np.logical_and(mask_Tim, mask_norv) # A star is a member in both analyses
    mask_none = np.logical_and(~mask_Tim, ~mask_norv) # A star is a NONmember in both analyses
    mask_Timyes_Marusano = np.logical_and(mask_Tim, ~mask_norv)
    mask_Timno_Marusayes = np.logical_and(~mask_Tim, mask_norv)

    print('Both', len(dall[mask_both]))
    print('None', len(dall[mask_none]))
    print('Timyes Marusano', len(dall[mask_Timyes_Marusano]))
    print('Timno Marusayes', len(dall[mask_Timno_Marusayes]))


    # Compare the probabilities of the non-bg component with the highest probability

def prepare_Tims_data():
    memb_usco = np.load('usco_res/final_membership.npy')
    data_usco = Table.read('usco_res/usco_run_subset.fit')
    for i in range(memb_usco.shape[1] - 1):
        data_usco['Comp_USco_%d' % (i + 1)] = memb_usco[:, i]
    data_usco['Comp_bg_USco'] = memb_usco[:, -1]
    data_usco['nonbg'] = -(data_usco['Comp_bg_USco']-1.0)

    memb_ucl = np.load('ucl_res/final_membership.npy')
    data_ucl = Table.read('ucl_res/ucl_run_subset.fit')
    for i in range(memb_ucl.shape[1] - 1):
        data_ucl['Comp_UCL_%d' % (i + 1)] = memb_ucl[:, i]
    data_ucl['Comp_bg_UCL'] = memb_ucl[:, -1]
    data_ucl['nonbg'] = -(data_ucl['Comp_bg_UCL'] - 1.0)

    """
    memb_lcc = np.load('lcc_res/final_membership.npy')
    data_lcc = Table.read('lcc_res/lcc_run_subset.fit')
    for i in range(memb_ucl.shape[1] - 1):
        data_lcc['Comp_LCC_%d' % (i + 1)] = memb_lcc[:, i]
    data_lcc['Comp_bg_LCC'] = memb_lcc[:, -1]
    data_lcc['nonbg'] = -(data_lcc['Comp_bg_LCC'] - 1.0)
    """

    data_memb = vstack([data_usco, data_ucl])
    #data_memb = vstack([data_memb, data_lcc])

    # Find the highest probability value
    nonbg_usco = -(data_memb['Comp_bg_USco']-1.0)
    nonbg_ucl = -(data_memb['Comp_bg_UCL']-1.0)
    #nonbg_lcc = -(data_memb['Comp_bg_LCC']-1.0)

    data_memb['nonbg_USco'] = nonbg_usco
    data_memb['nonbg_UCL'] = nonbg_ucl
    #data_memb['nonbg_LCC'] = nonbg_lcc

    return data_memb

def compare_membership_probabilities(d):
    tim = prepare_Tims_data()
    d = join(d, tim, keys='source_id')

    d['nonbg_probability'] = -(d['comp_overlap_bg']-1.0)

    print 'd'
    print d


    # Members in the overall data
    mask_members_d = d['comp_overlap_1'] > 1  # Everything is False
    for i in range(1, 15 + 1):
        mask_members_d = np.logical_or(mask_members_d, d['comp_overlap_%d' % i] > 0.5)



    mask_members_usco = dall['Comp_USco_1'] > 1  # Everything is False
    for i in range(1, 4 + 1):
        mask_members_usco = np.logical_or(mask_members_usco, dall['Comp_USco_%d' % i] > 0.5)

    mask_members_ucl = dall['Comp_UCL_1'] > 1  # Everything is False
    for i in range(1, 4 + 1):
        mask_members_ucl = np.logical_or(mask_members_ucl, dall['Comp_UCL_%d' % i] > 0.5)

    mask_members = dall['Comp_UCL_1'] > 1  # Everything is False
    for i in range(1, 16 + 1):
        mask_members = np.logical_or(mask_members, dall['comp_overlap_%d' % i] > 0.5)


    import matplotlib.pyplot as plt

    fig=plt.figure()
    ax=fig.add_subplot(111)

    ax.scatter(d9rv['Z'], d9rv['W'], s=1, c='k')
    ax.scatter(d9norv['Z'], d9norv['W'], s=1, c='r')

    plt.savefig('compare.png')

    plt.show()

#compare_membership_probabilities_of_stars_with_and_without_radial_velocities(d)
compare_membership_probabilities(d)