#! /usr/bin/env python
"""
Plot many hexplots
"""

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not imported")
    pass

from distutils.dir_util import mkpath
from multiprocessing import Pool
from itertools import product
import logging
import numpy as np
import os
import sys
import pdb

"""
sys.path.insert(0, '..')
"""
ages = [5, 15, 30, 50]  #Myr
spreads = [1, 2, 5, 10] #pc
v_disps = [1, 2, 5, 10] #km/s
sizes = [25, 50, 100, 200] #nstars
precs = ['perf', 'half', 'gaia', 'double']
# ages = [5, 30]
# spreads = [5]
# v_disps = [2]
# sizes   = [25, 100]
# precs = ['perf', 'gaia', 'double']
"""
# DIFFERENT SERVERS GET DIFFERENT AGE,
# Mash: 5
# Malice: 15
# Motley : 30, 50
ages = [5, 15, 30, 50]
spreads = [1, 5] #pc
v_disps = [2, 10] #km/s
sizes = [25, 100] #nstars
precs = ['perf', 'half', 'gaia']
"""
# # EVEN PARAMETERS
# ages = [5, 15, 30, 50]
# spreads = [2, 10] #pc
# v_disps = [1, 5] #km/s
# sizes = [50, 200] #nstars
# precs = ['perf', 'half', 'gaia']

#precs_string = str(precs).strip("[]").replace(',','').replace("'", '')
precs_string = ' '.join(precs)
prec_val = {'perf': 1e-5, 'half':0.5, 'gaia': 1.0, 'double': 2.0}

def plotTheThing(scenario):
    for prec in precs:
        os.system("python plot_hexplot_synth.py {} {} {} {} ".\
            format(*scenario) + precs_string)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filemode='w',
                        filename="hexplot_many.log")
    logging.info("mash is performing many fits:\nages: {}\nspreads: {}\n"
                 "v_disps: {}\nsizes: {}\nprecs: {}\n".format(ages, spreads,
                                                              v_disps, sizes,
                                                              precs))
    # read in system arguments
    if len(sys.argv) > 1:
        ncpus = int(sys.argv[1])
    else:
        ncpus = 1
    logging.info("ncpus: {}".format(ncpus))
    scenarios = product(ages, spreads, v_disps, sizes)
    if ncpus > 1:
        p = Pool(ncpus)
        p.map(plotTheThing, scenarios)
    else:
        map(plotTheThing, scenarios)
        

