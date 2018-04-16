#! /usr/bin/env python
"""
Simple script to encapsulate the behaviour of generating a traceback file
for a pkl file with only an astropy astrometry table stored. Currently this
script is predominatly used to generate tb file for the synthesised data
sets.
"""

import chronostar.tracingback as tb
import pickle
import sys
import numpy as np

args = sys.argv[1:]
for infile in args:
    print("Working with: {} ...........".format(infile))
    savefile = "data/tb_" + infile[5:]
    
    t = pickle.load(open(infile,'r'))
    times = np.linspace(0,50,60)
    tb.traceback(t,times,savefile=savefile)
