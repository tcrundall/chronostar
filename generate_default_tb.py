#! /usr/bin/env python
"""
Simple script to encapsulate the behaviour of generating a traceback file
for a pkl file with only an astropy astrometry table stored. Currently this
script is predominatly used to generate tb file for the synthesised data
sets.
"""

import chronostar.traceback as tb
import pickle
import argparse
import sys
import numpy as np

args = sys.argv[1:]
for infile in args:
    print("Working with: {} ...........".format(infile))
# 
# parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--infile', dest='i', 
#                     help='astropy table of astrometry')
# args = parser.parse_args()
# 
    savefile = "data/tb_" + infile[5:]
    
    t = pickle.load(open(infile,'r'))
    times = np.linspace(0,30,60)
    tb.traceback(t,times,savefile=savefile)
