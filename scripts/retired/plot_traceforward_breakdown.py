from __future__ import division, print_function
"""
Generates plots of traceforward of a synthetic association at
different time steps to help reader visualise what is happening
to the stars in the various phase planes.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
import chronostar.traceorbit as torb
import chronostar.retired2.datatool as dt

rdir = "../results/synth_fit/50_2_1_50/"
# rdir = "../results/synth_fit/30_5_2_100/"

xyzuvw_init_file = rdir + "xyzuvw_init.npy"

xyzuvw_init = np.load(xyzuvw_init_file)
origin = np.load(rdir + 'origins.npy').item()
origin = dt.loadGroups(rdir + 'origins.npy')
max_age = origin.age
ntimes = int(max_age) + 1
#ntimes = 3
times = np.linspace(1e-5, max_age, ntimes)

traceforward = torb.trace_many_cartesian_orbit(xyzuvw_init, times, False)
nstars = xyzuvw_init.shape[0]

def plot_subplot(traceforward, t_ix, dim1, dim2, ax):
    flat_tf = traceforward.reshape(-1,6)
#    mins = np.min(flat_tf, axis=0)
#    maxs = np.max(flat_tf, axis=0)
    labels = ['X [pc]', 'Y [pc]', 'Z [pc]',
              'U [km/s]', 'V [km/s]', 'W [km/s]']

    for i in range(nstars):
        # plot orbits
        ax.plot(traceforward[i, :t_ix, dim1],
                 traceforward[i, :t_ix, dim2],
                 'b',
                 alpha=0.1)
        # plot current position
        ax.plot(traceforward[i, t_ix-1, dim1],
                 traceforward[i, t_ix-1, dim2],
                 'b.')
#    ax.set_xlim(mins[dim1], maxs[dim1])
#    ax.set_ylim(mins[dim2], maxs[dim2])
    ax.set_xlabel(labels[dim1])
    #ax.set_ylabel(labels[dim2])

def plot_row(traceforward, dim1, dim2, axs, row_ix):
    labels = ['X [pc]', 'Y [pc]', 'Z [pc]',
              'U [km/s]', 'V [km/s]', 'W [km/s]']
    axs[row_ix,0].set_ylabel(labels[dim2])
    plot_subplot(traceforward, 1, dim1, dim2, axs[row_ix, 0])
    plot_subplot(traceforward, int(ntimes / 2), dim1, dim2, axs[row_ix, 1])
    plot_subplot(traceforward, ntimes, dim1, dim2, axs[row_ix, 2])

    #axs[row_ix, 0].set_xlim(axs[row_ix, 2].get_xlim())
    #axs[row_ix, 0].set_ylim(axs[row_ix, 2].get_ylim())

    #axs[row_ix, 1].set_xlim(axs[row_ix, 2].get_xlim())
    #axs[row_ix, 1].set_ylim(axs[row_ix, 2].get_ylim())

plt.clf()
nrows = 3
fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(10,9),
                        sharey='row')#, sharey=True)
axs[0,0].set_title("{} Myr".format(int(times[0])))
axs[0,1].set_title("{} Myr".format(int(times[int(ntimes/2)])))
axs[0,2].set_title("{} Myr".format(int(times[-1])))
plot_row(traceforward, 0, 1, axs, 0)
plot_row(traceforward, 1, 2, axs, 1)
plot_row(traceforward, 3, 4, axs, 2)

plt.savefig("temp_plots/break_down_unmixed2.pdf")

plt.clf()
nrows = 3
fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(10,9),
                        sharey='row', sharex='row')#, sharey=True)
fig.set_tight_layout(True)
axs[0,0].set_title("{} Myr".format(int(times[0])))
axs[0,1].set_title("{} Myr".format(int(times[int(ntimes/2)])))
axs[0,2].set_title("{} Myr".format(int(times[-1])))
plot_row(traceforward, 0, 3, axs, 0)
plot_row(traceforward, 1, 4, axs, 1)
plot_row(traceforward, 2, 5, axs, 2)

plt.savefig("temp_plots/break_down_mixed2.pdf")

#times = np.array([1e-5])

#for i in range(nstars):
#    plt.plot(traceforward[i,:,0], traceforward[i,:,1], 'b')
#
# plt.xlabel('X [pc]')
# plt.ylabel('Y [pc]')
# plt.savefig("xy.png")
# plt.clf()
#
# # TRACEBACK
# for t_ix in range(times.shape[0]):
#         plt.clf()
#         for i in range(nstars):
#             plt.plot(traceforward[i,-t_ix-1:,0], traceforward[i,-t_ix-1:,3], 'b',
#                      alpha =0.3)
#             plt.plot(traceforward[i,-t_ix-1,0], traceforward[i,-t_ix-1,3], 'b.')
#         plt.title("{:2} Myr".format(int(t_ix)))
# #        plt.xlim(-320,450)
# #        plt.ylim(-35,35)
#         plt.xlabel('X [pc]')
#         plt.ylabel('U [km/s]')
#         plt.savefig("temp_plots/{:02}_xu_tb.png".format(t_ix))
#
# for t_ix in range(times.shape[0]):
#         plt.clf()
#         for i in range(nstars):
#             plt.plot(traceforward[i,-t_ix-1:,1], traceforward[i,-t_ix-1:,4], 'b',
#                      alpha =0.3)
#             plt.plot(traceforward[i,-t_ix-1,1], traceforward[i,-t_ix-1,4], 'b.')
#         plt.title("{:2} Myr".format(int(t_ix)))
# #        plt.xlim(-320,450)
# #        plt.ylim(-35,35)
#         plt.xlabel('Y [pc]')
#         plt.ylabel('V [km/s]')
#         plt.savefig("temp_plots/{:02}_yv_tb.png".format(t_ix))
#
# for t_ix in range(times.shape[0]):
#         plt.clf()
#         for i in range(nstars):
#             plt.plot(traceforward[i,-t_ix-1:,2], traceforward[i,-t_ix-1:,5], 'b',
#                      alpha =0.3)
#             plt.plot(traceforward[i,-t_ix-1,2], traceforward[i,-t_ix-1,5], 'b.')
#         plt.title("{:2} Myr".format(int(t_ix)))
#         plt.xlim(-320,450)
#         plt.ylim(-35,35)
#         plt.xlabel('Z [pc]')
#         plt.ylabel('W [km/s]')
#         plt.savefig("temp_plots/{:02}_zw_tb.png".format(t_ix))
#
# # TRACEFORWARD
# for t_ix in range(times.shape[0]):
#         plt.clf()
#         for i in range(nstars):
#             plt.plot(traceforward[i,:t_ix+1,0], traceforward[i,:t_ix+1,3], 'b',
#                      alpha =0.3)
#             plt.plot(traceforward[i,t_ix,0], traceforward[i,t_ix,3], 'b.')
#         plt.title("{:2} Myr".format(int(t_ix)))
# #        plt.xlim(-320,450)
# #        plt.ylim(-35,35)
#         plt.xlabel('X [pc]')
#         plt.ylabel('U [km/s]')
#         plt.savefig("temp_plots/{:02}_xu_tf.png".format(t_ix))
#
# for t_ix in range(times.shape[0]):
#         plt.clf()
#         for i in range(nstars):
#             plt.plot(traceforward[i,:t_ix+1,1], traceforward[i,:t_ix+1,4], 'b',
#                      alpha =0.3)
#             plt.plot(traceforward[i,t_ix,1], traceforward[i,t_ix,4], 'b.')
#         plt.title("{:2} Myr".format(int(t_ix)))
# #        plt.xlim(-320,450)
# #        plt.ylim(-35,35)
#         plt.xlabel('Y [pc]')
#         plt.ylabel('V [km/s]')
#         plt.savefig("temp_plots/{:02}_yv_tf.png".format(t_ix))
#
# for t_ix in range(times.shape[0]):
#         plt.clf()
#         for i in range(nstars):
#             plt.plot(traceforward[i,:t_ix+1,2], traceforward[i,:t_ix+1,5], 'b',
#                      alpha =0.3)
#             plt.plot(traceforward[i,t_ix,2], traceforward[i,t_ix,5], 'b.')
#         plt.title("{:2} Myr".format(int(t_ix)))
#         plt.xlim(-320,450)
#         plt.ylim(-35,35)
#         plt.xlabel('Z [pc]')
#         plt.ylabel('W [km/s]')
#         plt.savefig("temp_plots/{:02}_zw_tf.png".format(t_ix))
#

