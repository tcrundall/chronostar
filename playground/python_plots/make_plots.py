#! /usr/bin/env python
"""
Demo script for generating multiple plots and combinging them into movies.
e.g. "projecting" a normal distribution backward in time, generating a bunch
of plots with the same axes, then combing into a movie
"""
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):
    return 1./(sigma * np.sqrt(2*np.pi)) *\
            np.exp(-0.5*( (x-mu)/(sigma) )**2)

def curve(xs, mu, sigma):
   return gaussian(xs, mu, sigma)

nframes = 50
for i in range(0,nframes):
    mu = i*10.0/nframes
    sigma = 3.0*i/nframes
    xs = np.linspace(-10, 10, 100)
    plt.clf()
    plt.plot(xs, gaussian(xs, i*10.0/nframes, sigma))
    plt.savefig("plot_" + str(i) +".jpg")
