"""
Prepare known Sco-Cen members with known radial velocities.
Make two sets: One with radial velocities and one with very large radial velocity errors.
"""

import numpy as np
from astropy.table import Table, vstack, join
import matplotlib.pyplot as plt

data = Table.read('../data_table_cartesian_including_tims_stars_with_bg_ols_and_component_overlaps.fits')
datanorv = Table.read('scocen_members_with_artificially_broken_radial_velocities_for_comparison_with_all_tims_members_with_component_probabilities.fits')

print len(data), len(datanorv)

#mask = np.in1d(data['source_id'], datanorv['source_id'])
#data=data[mask]

d = join(datanorv, data, keys='source_id')

# membership by component
for i in range(1, 15+1):
    mask1 = d['comp_overlap_%d_1' % i]>0.5
    mask2 = d['comp_overlap_%d_2' % i]>0.5
    try:
        f=float(len(d[mask1]))/float(len(d[mask2]))
    except:
        f=None
    print i, len(d[mask2]), len(d[mask1]), f
mask1 = d['comp_overlap_bg_1']>0.5
mask2 = d['comp_overlap_bg_2']>0.5
print 'bg', len(d[mask2]), len(d[mask1]), float(len(d[mask1]))/float(len(d[mask2]))


fig=plt.figure()
for i in range(1, 15+1):
    ax=fig.add_subplot(4,4,i)
    ax.scatter(d['comp_overlap_%d_1'%i], d['comp_overlap_%d_2'%i], s=1)
    ax.axhline(y=0.5, linewidth=0.5, color='k')
    ax.axvline(x=0.5, linewidth=0.5, color='k')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

ax=fig.add_subplot(4,4,16)
ax.scatter(d['comp_overlap_bg_1'], d['comp_overlap_bg_2'], s=1)
ax.axhline(y=0.5, linewidth=0.5, color='k')
ax.axvline(x=0.5, linewidth=0.5, color='k')
ax.set_xlabel('p noRV')
ax.set_ylabel('p withRV')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


b=20
fig=plt.figure()
ax=fig.add_subplot(111)
fig.suptitle('p(bg)')
ax.hist(d['comp_overlap_bg_1'], histtype='step', color='r', bins=b, label='No RV')
ax.hist(d['comp_overlap_bg_2'], histtype='step', color='k', bins=b, label = 'RV')
ax.legend()

mask = d['comp_overlap_bg_2']<0.5
masknorv = d['comp_overlap_bg_1']<0.5
print('NON-Background:', 'with rv', len(d[mask]), 'noRV', len(d[masknorv]), float(len(d[masknorv]))/float(len(d[mask])))

plt.show()