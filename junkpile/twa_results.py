"""Unimaginative script for generating plots of TWA fixed age fits and their
sizes w.r.t time.
"""
#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

widths = np.array([
    [13.15, 1.12, 1.09],
    [12.51,1.09,0.94],
    [12.09, 1.15, 0.97],
    [11.64, 1.08,0.92],
    [11.55,1.14,0.97],
    [11.54, 1.11, 0.95],
    [11.71,1.10,0.99], #6.0
    [11.96,1.16,1.04],
    [12.64,1.21,1.16],
    [12.94,1.23,1.16],
    [13.73,1.34,1.31],
    [14.17,1.51,1.32],
    [14.95,1.42,1.22],
    ])

c_widths = np.array([
    [10.91, 1.78, 1.40],
    [10.77, 1.57, 1.47],
    [10.18, 1.61, 1.44],
    [ 9.26, 1.72, 1.30],
    [ 8.69, 1.57, 1.31],
    [ 8.36, 1.59, 1.51],
    [ 7.89, 1.59, 1.39],
    [ 7.86, 1.64, 1.36],
    [ 7.81, 1.58, 1.34],
    [ 7.91, 1.53, 1.24],
    [ 8.34, 1.74, 1.50],
    [ 8.51, 1.85, 1.47],
    [ 8.85, 1.76, 1.49],
    ])

ages = np.linspace(0,12,13)

plt.errorbar(ages, c_widths[:,0], yerr=[c_widths[:,2], c_widths[:,1]])
plt.ylim(ymin=0)
plt.ylabel("Radius of fit [pc]")
plt.xlabel("Age [Myr]")
plt.title("All TWA Stars")
plt.savefig("plots/twa_plots/TWA_fit.eps")
#plt.show()
plt.clf()

plt.errorbar(ages, widths[:,0], yerr=[widths[:,2], widths[:,1]], label="all")
plt.errorbar(ages, c_widths[:,0], yerr=[c_widths[:,2], c_widths[:,1]],
             label='core')
plt.ylim(ymin=0)
plt.ylabel("Radius of fit [pc]")
plt.xlabel("Age [Myr]")
plt.title("Size of volume occupied by TWA Stars over Time")
plt.legend(loc='best')
plt.savefig("plots/twa_plots/TWA_both_fit.eps")
#plt.show()
