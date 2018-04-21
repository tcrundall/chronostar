#!/usr/bin/env python -W ignore
"""
test_expectmax
-----------------------------

Tests for `expectmax` module
"""
from __future__ import division, print_function

import os.path
import sys
import tempfile
import unittest

sys.path.insert(0, '..')  # hacky way to get access to module

import chronostar.retired.expectmax as em
import chronostar.retired.synthesiser as syn
import chronostar.retired.tracingback as tb
import chronostar.retired.groupfitter as gf
from chronostar.retired import utils
import numpy as np
import pickle


class ExpectmaxTestCase(unittest.TestCase):
    def setUp(self):
        self.groups_pars_ex = np.array([
            # X, Y, Z, U, V, W, dX, dY, dZ,dV,Cxy,Cxz,Cyz,age,nstars
            [ 0, 0, 0, 0, 0, 0, 10, 10, 10, 5, -.3, -.6, .2, 10, 500],
            [50,50, 0, 0, 0, 0, 10, 10, 10, 5,  .1,  .3, .2, 10, 500],
        ])

        self.tempdir = tempfile.mkdtemp()
        self.synth_file = os.path.join(self.tempdir, 'synth_data.pkl')
        self.tb_file = os.path.join(self.tempdir, 'tb_data.pkl')

    def tearDown(self):
        try:
            os.remove(self.synth_file)
        except OSError:
            pass
        try:
            os.remove(self.tb_file)
        except OSError:
            pass
        os.rmdir(self.tempdir)


    @unittest.skip("not yet implemented")
    def test_calc_errors(self):
        means = np.array([10,50,100,10,10,10, .2, .2, .2,  .1,  0 , 0,  0,10])
        stds  = np.array([ 2, 5, 10, 2, 2, 2,.05,.05,.05,.025,.05,.05,.05, 1])


    def test_maximisation(self):
        """
        Synthesise a tb file with negligible error, retrieve initial
        parameters
        """
        ngroups = self.groups_pars_ex.shape[0]
        nstars = np.sum(self.groups_pars_ex[:,-1])

        groups_pars_in = utils.internalise_multi_pars(self.groups_pars_ex)

        # neligible error, anything smaller runs into problems with matrix
        # inversions
        error = 1e-5
        ntimes = 20

        tb_file = "tmp_expectmax_tb_file.pkl"

        # to save time, check if tb_file is already created
        try:
            with open(tb_file):
                pass
        # if not created, then create it. Careful though! May not be the same
        # as group_pars. So if test fails try deleting tb_file from
        # directory
        except IOError:
            # generate synthetic data
            syn.synthesise_data(
                ngroups, self.groups_pars_ex, error, savefile=self.synth_file
            )
            with open(self.synth_file, 'r') as fp:
                t = pickle.load(fp)

            max_age = np.max(groups_pars_in[:,-1])
            times = np.linspace(0, 2*max_age, ntimes)
            tb.traceback(t, times, savefile=tb_file)

        # Initialse z to have first 100 stars associated with group1
        # and the second 100 stars associated with group2
        # CURRENTLY HARDCODED TO ONLY DEAL WITH 2 GROUPS
        z = np.zeros((int(nstars), int(ngroups)))
        nstars_1 = int(self.groups_pars_ex[0,-1])
        z[:nstars_1,0] = 1
        z[nstars_1:,1] = 1

        # find best fit
        best_fits, _, _ = em.maximise(
            tb_file, ngroups, z=z, burnin_steps=200, sampling_steps=200,
            init_conditions=groups_pars_in
        )


        for ctr, (best_fit, group_pars_ex) in\
                enumerate(zip(best_fits, self.groups_pars_ex)):
            means = best_fit[0:6]
            stds = 1 / best_fit[6:10]
            corrs = best_fit[10:13]
            age = best_fit[13]

            tol_mean = 3.5
            tol_std = 2.5
            tol_corr = 0.3
            tol_age = 0.5

            self.assertTrue(
                np.max(abs(means - group_pars_ex[0:6])) < tol_mean,
                msg="\nFailed {} received:\n{}\nshould be within {} to:\n{}".
                    format(ctr, means, tol_mean, group_pars_ex[0:6]))
            self.assertTrue(
                np.max(abs(stds - group_pars_ex[6:10])) < tol_std,
                msg="\nFailed {} received:\n{}\nshould be close to:\n{}".
                    format(ctr, stds, tol_std, group_pars_ex[6:10]))
            self.assertTrue(
                np.max(abs(corrs - group_pars_ex[10:13])) < tol_corr,
                msg="\nFailed {} received:\n{}\nshould be close to:\n{}". \
                format(ctr, corrs, tol_corr, group_pars_ex[10:13]))
            self.assertTrue(
                np.max(abs(age - group_pars_ex[13])) < tol_age,
                msg="\nFailed {} received:\n{}\nshould be close to:\n{}". \
                format(ctr, age, tol_age, group_pars_ex[13]))

    def test_expectation(self):
        ngroups = self.groups_pars_ex.shape[0]
        nstars = np.sum(self.groups_pars_ex[:, -1])

        groups_pars_in = utils.internalise_multi_pars(self.groups_pars_ex)

        # neligible error - smaller vals lead to problems with matrix inversions
        error = 1e-5
        ntimes = 20

        tb_file = "tmp_expectmax_tb_file.pkl"

        # to save time, check if tb_file is already created
        try:
            with open(tb_file):
                pass
        # if not created, then create it. Careful though! May not be the same
        # as group_pars. So if test fails try deleting tb_file from
        # directory
        except IOError:
            # generate synthetic data
            syn.synthesise_data(
                ngroups, self.groups_pars_ex, error, savefile=self.synth_file
            )
            with open(self.synth_file, 'r') as fp:
                t = pickle.load(fp)

            max_age = np.max(groups_pars_in[:, -1])
            times = np.linspace(0, 2 * max_age, ntimes)
            tb.traceback(t, times, savefile=tb_file)

        star_pars = gf.read_stars(tb_file)

        z = em.expectation(star_pars, groups_pars_in)

        # check membership list totals to nstars in group
        self.assertTrue(np.isclose(np.sum(z), nstars))
        self.assertTrue(np.allclose(np.sum(z, axis=1), 1.0))
        self.assertTrue(
            np.allclose(np.sum(z, axis=0), self.groups_pars_ex[:,-1], atol=0.1)
        )

        nstars1 = int(self.groups_pars_ex[0,-1])
        nstars2 = int(self.groups_pars_ex[1,-1])
        self.assertTrue( (z[:nstars1,0] > z[:nstars1,1]).all() )
        self.assertTrue( (z[nstars1:,0] < z[nstars1:,1]).all() )


def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(ExpectmaxTestCase)
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
