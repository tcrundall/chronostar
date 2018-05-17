# coding: utf-8
#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../..')
import chronostar.synthesiser as syn
import chronostar.groupfitter as gf
import chronostar.traceorbit as torb

xyzuvw_init = np.load('xyzuvw_init.npy')
origin = np.load('origins.npy').item()
max_age = origin.age
ntimes = int(max_age) + 1
times = np.linspace(1e-5, max_age, ntimes)

traceback = torb.traceManyOrbitXYZUVW(xyzuvw_init, times, False)
nstars = xyzuvw_init.shape[0]

#times = np.array([1e-5])

for i in range(nstars):
    plt.plot(traceback[i,:,0], traceback[i,:,1], 'b')

plt.xlabel('X [pc]')
plt.ylabel('Y [pc]')
plt.savefig("xy.png")
plt.clf()

# TRACEBACK
for t_ix in range(times.shape[0]):
        plt.clf()
        for i in range(nstars):
            plt.plot(traceback[i,-t_ix-1:,0], traceback[i,-t_ix-1:,3], 'b',
                     alpha =0.3)
            plt.plot(traceback[i,-t_ix-1,0], traceback[i,-t_ix-1,3], 'b.')
        plt.title("{:2} Myr".format(int(t_ix)))
#        plt.xlim(-320,450)
#        plt.ylim(-35,35)
        plt.xlabel('X [pc]')
        plt.ylabel('U [km/s]')
        plt.savefig("temp_plots/{:02}_xu_tb.png".format(t_ix))

for t_ix in range(times.shape[0]):
        plt.clf()
        for i in range(nstars):
            plt.plot(traceback[i,-t_ix-1:,1], traceback[i,-t_ix-1:,4], 'b',
                     alpha =0.3)
            plt.plot(traceback[i,-t_ix-1,1], traceback[i,-t_ix-1,4], 'b.')
        plt.title("{:2} Myr".format(int(t_ix)))
#        plt.xlim(-320,450)
#        plt.ylim(-35,35)
        plt.xlabel('Y [pc]')
        plt.ylabel('V [km/s]')
        plt.savefig("temp_plots/{:02}_yv_tb.png".format(t_ix))

for t_ix in range(times.shape[0]):
        plt.clf()
        for i in range(nstars):
            plt.plot(traceback[i,-t_ix-1:,2], traceback[i,-t_ix-1:,5], 'b',
                     alpha =0.3)
            plt.plot(traceback[i,-t_ix-1,2], traceback[i,-t_ix-1,5], 'b.')
        plt.title("{:2} Myr".format(int(t_ix)))
        plt.xlim(-320,450)
        plt.ylim(-35,35)
        plt.xlabel('Z [pc]')
        plt.ylabel('W [km/s]')
        plt.savefig("temp_plots/{:02}_zw_tb.png".format(t_ix))

# TRACEFORWARD
for t_ix in range(times.shape[0]):
        plt.clf()
        for i in range(nstars):
            plt.plot(traceback[i,:t_ix+1,0], traceback[i,:t_ix+1,3], 'b',
                     alpha =0.3)
            plt.plot(traceback[i,t_ix,0], traceback[i,t_ix,3], 'b.')
        plt.title("{:2} Myr".format(int(t_ix)))
#        plt.xlim(-320,450)
#        plt.ylim(-35,35)
        plt.xlabel('X [pc]')
        plt.ylabel('U [km/s]')
        plt.savefig("temp_plots/{:02}_xu_tf.png".format(t_ix))

for t_ix in range(times.shape[0]):
        plt.clf()
        for i in range(nstars):
            plt.plot(traceback[i,:t_ix+1,1], traceback[i,:t_ix+1,4], 'b',
                     alpha =0.3)
            plt.plot(traceback[i,t_ix,1], traceback[i,t_ix,4], 'b.')
        plt.title("{:2} Myr".format(int(t_ix)))
#        plt.xlim(-320,450)
#        plt.ylim(-35,35)
        plt.xlabel('Y [pc]')
        plt.ylabel('V [km/s]')
        plt.savefig("temp_plots/{:02}_yv_tf.png".format(t_ix))

for t_ix in range(times.shape[0]):
        plt.clf()
        for i in range(nstars):
            plt.plot(traceback[i,:t_ix+1,2], traceback[i,:t_ix+1,5], 'b',
                     alpha =0.3)
            plt.plot(traceback[i,t_ix,2], traceback[i,t_ix,5], 'b.')
        plt.title("{:2} Myr".format(int(t_ix)))
        plt.xlim(-320,450)
        plt.ylim(-35,35)
        plt.xlabel('Z [pc]')
        plt.ylabel('W [km/s]')
        plt.savefig("temp_plots/{:02}_zw_tf.png".format(t_ix))


