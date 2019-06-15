'''
Explore fits and members of scocen
'''

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '..')

from chronostar.component import SphereComponent
from chronostar import tabletool
from chronostar import likelihood
from chronostar import expectmax

component_file = '../results/all_nonbg_scocen_comps.npy'
membership_file = '../results/all_scocen_total_membership.npy'
joined_table = '../data/scocen/joined_scocen_no_duplicates.fit'

star_pars = tabletool.build_data_dict_from_table(joined_table,
                                                 historical=True)
all_comps = SphereComponent.load_raw_components(component_file)
init_z = np.load(membership_file)

# pop manually determined duplicates
if True:
    all_comps.pop(9)
    all_comps.pop(6)
    init_z = init_z[(np.array([0,1,2,3,4,5,7,8]),)]

print(len(all_comps))
print(len(init_z))
init_z.shape = (1,-1)

memberships = expectmax.expectation(star_pars, all_comps,
                                    old_memb_probs=init_z)
members_mask = np.where(memberships[:,-1] < 0.5)
members_prob = memberships[members_mask]

mns = star_pars['means']
labels = 'XYZUVW'
units = 3*['pc'] + 3*['km/s']

# for dim1, dim2 in [(0,1), (0,2), (0,3), (1,4), (2,5), (3,4)]:
for dim1, dim2 in [(0,3)]:
    plt.clf()
    plt.plot(mns[members_mask,dim1], mns[members_mask,dim2],
             '.', color='blue', alpha=0.2)
    [c.plot(dim1=dim1, dim2=dim2) for c in all_comps]
    plt.xlabel('{} [{}]'.format(labels[dim1], units[dim1]))
    plt.ylabel('{} [{}]'.format(labels[dim2], units[dim2]))
    plt.savefig('../plots/scocen_{}{}.pdf'.format(labels[dim1],
                                                  labels[dim2]))
