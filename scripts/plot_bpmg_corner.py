from __future__ import print_function, division

"""
Teensy script to plot corner plots for BPMG fits
"""

import numpy as np
import matplotlib.pyplot as plt
import corner

chainfiles = [
    '../results/em_fit/beta_Pic_solo_results/final_chain.npy',
    '../results/em_fit/beta_Pic_results/group0/final_chain.npy',
]

labels = [
    'X [pc]',
    'Y [pc]',
    'Z [pc]',
    'U [km/s]',
    'V [km/s]',
    'W [km/s]',
    r'$\sigma_{xyz}$',
    r'$\sigma_{uvw}$',
    't [Myr]',
]

plot_names = ['bpmg_solo_corner.pdf', 'bpmg_corner.pdf']
rev_flags = [True, False]

for chainfile, plot_name, rev_flag in zip(chainfiles, plot_names,
                                           rev_flags):
    print("Plotting {}".format(plot_name))
    chain = np.load(chainfile).reshape(-1,9)
    chain[:,6:8] = np.exp(chain[:,6:8])
    # plt.tick_params(direction='in')
    fig = corner.corner(
        chain,
        labels=labels,
        # reverse=True,
        label_kwargs={'fontsize':'xx-large'},
        max_n_ticks=4,
    )
    print("Applying tick parameters")
    for ax in fig.axes:
        ax.tick_params(direction='in', labelsize='x-large', top=True,
                       right=True)
    print("... saving")
    plt.savefig('temp_plots/' + plot_name)
