#!/usr/bin/env python -W ignore

import logging
import numpy as np
import pdb
import sys
sys.path.insert(0,'..')

import chronostar.investigator as iv
import chronostar.quadplotter as qp


def investigate():

    #age = 7
    age = 5
    mock_twa_pars = [
        -80, 80, 50, 10, -20, -5, 5, 5, 5, 10, 0.0, 0.0, 0.0, age, 50
    ]
    maxtime = 10
    times = np.linspace(0,maxtime,11)
    nfixed_fits = 21

    test_save = SAVE_DIR + 'test.npy'
    np.save(test_save, mock_twa_pars)
    stored_pars = np.load(test_save)
    logging.info('Retrieved: {}'.format(stored_pars))

    logging.info('__Initialising__')

    my_synth_fit = iv.SynthFit(
        init_group_pars=mock_twa_pars, save_dir=SAVE_DIR, times=times,
        nfixed_fits=nfixed_fits,
    )

    logging.info('__Investigating__')
    my_synth_fit.investigate(plot_it=True)
    save_file = SAVE_DIR + 'my_synth_fit.npy'
    logging.info('__Storing__')
    np.save(save_file, my_synth_fit)

def quadplot():
    logging.info('__Quadplotting__')
    save_file = SAVE_DIR + 'my_synth_fit.npy'
    stored_synth_fit = np.load(save_file).item()
    qp.quadplot_synth_res(stored_synth_fit, save_dir=SAVE_DIR)


SAVE_DIR = '../results/synth_results/basic/'
if __name__ == '__main__':
    logging.basicConfig(
        filename=SAVE_DIR+'qp_iv_demo.log', level=logging.DEBUG,
        filemode='w'
    )
    logging.info('----- Started -----')

    investigate()
    #pdb.set_trace()
    quadplot()
    logging.info('----- Finished -----')
