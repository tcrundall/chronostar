#!/usr/bin/env python -W ignore

import sys
sys.path.insert(0,'..')

import chronostar.investigator as iv
import pdb
import numpy as np
import logging
import pickle

#logging.basicConfig(
#    format='%(asctime)s %(message)s',
#    level=logging.DEBUG
#)


save_dir = '../data/synth_trial/'

logging.basicConfig(
    filename=save_dir+'investigator_demo.log', level=logging.DEBUG,
    filemode='w'
)
logging.info('----- Started -----')
mock_group_pars =\
    [20, 20, 5, -22, -10, -3, 5, 5, 5, 3, 0.5, 0.5, 0.3, 10, 10]
times = np.linspace(0,20,5)
fixed_ages = np.linspace(0,20,3)

logging.info('__Initialising__')
mysf = iv.SynthFit(
    mock_group_pars, times=times, nfixed_fits=3, save_dir=save_dir
)
logging.info('__Investigating__')
mysf.investigate()
save_file = save_dir + 'synth_result'
np.save(save_file, mysf)

logging.info('__Storing__')
yoursf = np.load(save_file+'.npy').item()
logging.info('fixed_ages attribute from storage: {}'.format(yoursf.fixed_ages))
logging.info('----- Finished -----')


