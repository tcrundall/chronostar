
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import sys
sys.path.insert(0, '..')
from chronostar.component import SphereComponent

def update_1D_lims(current_lims, recent_lims):
    if current_lims is None:
        return recent_lims
    else:
        try:
            return [np.min((current_lims[0], recent_lims[0])),
                    np.max((current_lims[1], recent_lims[1]))]
        except:
            import pdb; pdb.set_trace()


labels = 'XYZUVW'
units = 3*['pc'] + 3*['km/s']
plt_dir = '../plots/scocen_formation/'
all_comps_file = '../results/all_nonbg_scocen_comps.npy'

# Reading in and tidying components
all_comps = SphereComponent.load_raw_components(all_comps_file)

# # Pop off some manually identified duplicates
# all_comps.pop(9)
# all_comps.pop(6)

print('ages of components: {}'.format([c.get_age() for c in all_comps]))

max_age = np.max([c.get_age() for c in all_comps])

ntimes = 100
times = np.linspace(max_age, 0, ntimes)

#for time in times:

time_ix = ntimes - ntimes/2
time = times[int(ntimes/2)]


# for dim1, dim2 in [(0,2)]:
for dim1, dim2 in [(0,1), (0,2), (0,3), (1,4), (2,5), (1,2)]:
    lims = 6*[None]
    for time_ix, time in enumerate(times[::-1]):
        print('plot {:3}: time {:4.2f}'.format(time_ix, time))
        plt.clf()

        for c in all_comps:
            # only plot component if it exists
            # print('plotting {}'.format(c))
            if c.get_age() > time:
                # print('{} is greater than {}'.format(c.get_age(), time))
                c_copy = SphereComponent(c.get_pars())

                # modify age so that 'comp_now' is plotted at time `time`
                # time is how long ago, we want copy.age to be time since
                # birth, that is c.get_age - time
                c_copy.update_attribute({'age':c.get_age()-time})
                # print('pars updated to: {}'.format(c_copy.get_pars()))

                c_copy.plot(dim1=dim1, dim2=dim2, comp_now=True, comp_then=True,
                            comp_orbit=True)

                # There was a prior issue (
                # if c_copy.get_mean_now()[1] < -200:
                #     import pdb; pdb.set_trace()

        plt.xlim()
        plt.xlabel('{} [{}]'.format(labels[dim1], units[dim1]))
        plt.ylabel('{} [{}]'.format(labels[dim2], units[dim2]))

        lims[dim1] = update_1D_lims(lims[dim1], plt.xlim())
        lims[dim2] = update_1D_lims(lims[dim2], plt.ylim())

        plt.xlim(lims[dim1])
        plt.ylim(lims[dim2])

        plt.title('Traceback time {:5.1f} Myr'.format(time))
        plt.savefig(plt_dir + '{:03}_{}{}.png'.format(time_ix,
                                                      labels[dim1],
                                                      labels[dim2]))
