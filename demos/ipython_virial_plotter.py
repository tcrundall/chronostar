# coding: utf-8
import virial_plotter as vp
get_ipython().magic(u'cat virial_plotter.py')
MU
vp.MU
vp.SIG
np.exp
import numpy as np
vp.np
np.exp(3)
np.log(3)
np.log(4) - np.log(2)
SIG
vp.SIG
np.log(10) - np.log(1)
vp.lognormal
help(vp.lognormal)
vp.lognormal(1, np.log(1), np.log(10) - np.log(1))
myxs = np.linspace(0,10)
vp.lognormal(myxs, np.log(1), np.log(10) - np.log(1))
myxs = np.linspace(1e-5, 10)
import matplotlib.pyplot as plt
plt.plot(myxs, vp.lognormal(myxs np.log(1), np.log(10) - np.log(1)))
plt.plot(myxs, vp.lognormal(myxs, np.log(1), np.log(10) - np.log(1)))
plt.show()
plt.clf()
plt.plot(myxs, vp.lognormal(myxs, np.log(1), np.log(10) - np.log(1)))
plt.show()
plt.clf()
plt.plot(myxs, vp.lognormal(myxs, np.log(1), np.log(10) - np.log(1)))
plt.savefig("temp_plots/vp.png")
myxs = np.linspace(1e-5, 10, 100)
plt.clf()
plt.plot(myxs, vp.lognormal(myxs, np.log(3), np.log(10) - np.log(1)))
plt.savefig("temp_plots/vp.png")
plt.plot(myxs, vp.lognormal(myxs, np.log(3), 0.5))
plt.savefig("temp_plots/vp.png")
plt.plot(myxs, vp.lognormal(myxs, np.log(3), 1.))
plt.savefig("temp_plots/vp.png")
plt.plot(myxs, vp.lognormal(myxs, 1.05, 0.105))
plt.savefig("temp_plots/vp.png")
plt.plot(myxs, vp.lognormal(myxs, 1.05, np.sqrt(0.105)))
plt.savefig("temp_plots/vp.png")
main_mean = np.log(3)
mode = 3
stds = np.linspace(0.2,1.0,5)
stds
stds = np.array([1.,10.,10])
means = stds**2 + np.log(mode)
means
stds = np.linspace([1.,10.,10])
stds = np.linspace(1.,10.,10)
means = stds**2 + np.log(mode)
means
stds = np.linspace(1.,4,10)
means = stds**2 + np.log(mode)
means
plt.clf()
for mn, std in zip(means, stds):
    plt.plot(myxs, vp.lognormal(myxs, mn, std))
    
plt.savefig("temp_plots/vp.png")
for mn, std in zip(means, stds):
    plt.plot(myxs, vp.lognormal(myxs, mn, std)/np.max(vp.lognormal(myxs, mn, std)))
    
plt.clf()
for mn, std in zip(means, stds):
    plt.plot(myxs, vp.lognormal(myxs, mn, std)/np.max(vp.lognormal(myxs, mn, std)))
    
plt.savefig("temp_plots/vp.png")
stds
stds = np.linspace(0.7, 2,3)
plt.clf()
for std in stds:
    mn = std**2 + mode
    plt.plot(myxs, vp.lognormal(myxs, mn, std)/np.max(vp.lognormal(myxs, mn, std)))
    
plt.savefig("temp_plots/vp.png")
mode
for std in stds:
    mn = std**2 + np.log(mode)
    plt.plot(myxs, vp.lognormal(myxs, mn, std)/np.max(vp.lognormal(myxs, mn, std)))
    
plt.clf()
for std in stds:
    mn = std**2 + np.log(mode)
    plt.plot(myxs, vp.lognormal(myxs, mn, std)/np.max(vp.lognormal(myxs, mn, std)))
    
plt.savefig("temp_plots/vp.png")
plt.clf()
for std in stds:
    mn = std**2 + np.log(mode)
    plt.plot(myxs, vp.lognormal(myxs, mn, std)/np.max(vp.lognormal(myxs, mn, std)),
    label=r"$\sigma = ${:.2}".format(std))
    
plt.savefig("temp_plots/vp.png")
plt.legend(loc='best')
plt.savefig("temp_plots/vp.png")
stds = np.array([0.2, 0.7, 1.2, 1.7])
plt.clf()
for std in stds:
    mn = std**2 + np.log(mode)
    plt.plot(myxs, vp.lognormal(myxs, mn, std)/np.max(vp.lognormal(myxs, mn, std)),
    label=r"$\sigma = ${:.2}".format(std))
    
plt.legend(loc='best')
plt.xlabel(r"$\alpha$")
plt.savefig("temp_plots/vp.png")
plt.legend(loc=4)
plt.savefig("temp_plots/vp.png")
