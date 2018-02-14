#!/usr/bin/env python -W ignore
"""
investigator_test.py

Tests for `investigator` module.
"""

import logging
import numpy as np
import os
import pdb
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, '..')

import chronostar.investigator as iv
from chronostar import utils

class InvestigatorTestCase(unittest.TestCase):
    def setUp(self):
        #self.tempdir = tempfile.mkdtemp()
        self.tempdir = 'investigator_log/'
        self.mock_twa_pars = [
            -80, 80, 50, 10, -20, -5, 5, 5, 5, 2,
            0.0, 0.0, 0.0, 5, 40
        ]

#    def tearDown(self):
#        shutil.rmtree(self.tempdir)

    def get_nearest_age_ix(self, ages, true_age):
        return (np.abs(ages-true_age)).argmin()

    def test_initialisation(self):
        logging.basicConfig(
            filename=self.tempdir + 'investigator_test.log',
            level=logging.DEBUG,
            filemode='w',
            )
        logging.info('----- running test_initialisation -----')

        times = np.linspace(0,10,5)
        nfixed_fits = 3
        synthfit_file = self.tempdir + 'my_synthfit.npy'
        try:
            my_synthfit = np.load(synthfit_file).item()
        except IOError:
            my_synthfit = iv.SynthFit(
                self.mock_twa_pars, save_dir=self.tempdir,
                times=times, nfixed_fits=nfixed_fits
            )

            my_synthfit.investigate()
            np.save(synthfit_file, my_synthfit)

        # check shape of free fit data
        assert(my_synthfit.free_age_fit.mean_mu.shape == (6,))
        assert(my_synthfit.free_age_fit.mean_sigma.shape == (6,6))

        ta_ix = self.get_nearest_age_ix(
            my_synthfit.fixed_ages, my_synthfit.init_group_pars_ex[13],
        )
        assert(
            np.allclose(
                my_synthfit.fixed_age_fits[ta_ix].mean_mu,
                my_synthfit.init_group_pars_ex[:6], atol=3.5
            )
        )

        assert(
            np.allclose(
                utils.generate_pars(
                    my_synthfit.fixed_age_fits[ta_ix].mean_sigma
                )[:4],
                my_synthfit.init_group_pars_ex[6:10], atol=1.5
            )
        )

        true_bayes_spread = utils.approx_spread_from_sample(
            utils.internalise_pars(my_synthfit.init_group_pars_ex)
        )

        try:
            assert(np.isclose(
                true_bayes_spread, my_synthfit.fixed_age_fits[ta_ix].mean_radius,
                atol=0.7
            ))
        except AssertionError:
            pdb.set_trace()

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(InvestigatorTestCase)
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
