"""Dedicated script which generated an age PDF plot of TWA for honours thesis.
"""
#!/usr/bin/env python
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

tstamps =\
    [
    "0_0_114471",
    #"0_0_114261",
    "1_0_114262",
    "2_0_114304",
    "3_0_114261",
    "4_0_114261",
    "5_0_114260",
    "6_0_114258",
    "7_0_114261",
    "8_0_114261",
    "9_0_115034",
    "10_0_114302",
    "11_0_114303",
    "12_0_114303",
    ]

mn_lnprob = np.zeros(len(tstamps))
loc = "/short/kc5/results/"
loc = "results/"
for i, tstamp in enumerate(tstamps):
    filename = loc+"lnprob_"+tstamp+"_1_0.pkl"
    lnprobs, x, y, t = pickle.load(open(filename, 'r'))
    mn_lnprob[i] = np.mean(lnprobs)

for i, tstamp in enumerate(tstamps):
    filename = loc+"lnprob_"+tstamp+"_1_0.pkl"
    lnprobs, x, y, t = pickle.load(open(filename, 'r'))
    mn_lnprob[i] = np.mean(lnprobs)

times = np.linspace(0,12,13)
norm_lnprob = np.exp(mn_lnprob - np.max(mn_lnprob))
tnew = np.linspace(times.min(), times.max(), 300)
lnprob_smooth = spline(times,norm_lnprob,tnew)

plt.plot(tnew, lnprob_smooth)
plt.ylim((0, 1.1))
plt.xlabel("age [Myr]")
plt.ylabel("P(age)")
plt.savefig("plots/twa_age_pdf.eps")
