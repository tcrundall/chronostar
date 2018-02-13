#!/usr/bin/env python -W ignore

import sys
sys.path.insert(0,'..')

import chronostar.investigator as iv
import pdb
import numpy as np
import logging

#logging.basicConfig(
#    format='%(asctime)s %(message)s',
#    level=logging.DEBUG
#)



logging.basicConfig(filename='investigator_demo.log', level=logging.DEBUG)
logging.info('Started')
mock_group_pars =\
    [20, 20, 5, -22, -10, -3, 5, 5, 5, 3, 0.5, 0.5, 0.3, 10, 10]
times = np.linspace(0,20,5)
fixed_ages = np.linspace(0,20,3)

mysf = iv.SynthFit(mock_group_pars, times=times, fixed_ages=fixed_ages)
mysf.investigate()

pdb.set_trace()

logging.info('Finished')


