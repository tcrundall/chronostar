#!/usr/bin/env python -W ignore
"""
test_groupfitter
-----------------------------

Tests for `groupfitter` module
"""
from __future__ import division, print_function

from emcee.utils import MPIPool
import os.path
import sys
import tempfile
import unittest

sys.path.insert(0, '..')  # hacky way to get access to module

import chronostar.groupfitter as gf
import chronostar.synthesiser as syn
import chronostar.analyser as an
import chronostar.traceback as tb
import numpy as np
import pickle


class GroupfitterTestCase(unittest.TestCase):
    def setUp(self):
        self.times = np.array([0.0, 1.0, 2.0])

        self.xyzuvw = np.array([
            [
                [0.0, 0.5, 1.0, 1.1, 0.2, -1.0],
                [-1.0, 0.3, 2.0, 0.9, 0.2, -1.0],
                [-1.2, 0.1, 3.0, 0.7, 0.2, -1.0],
            ],
            [
                [0.0, 0.5, 1.0, 1.0, 0.2, -1.0],
                [0.0, 0.5, 1.0, 1.0, 0.2, -1.0],
                [0.0, 0.5, 1.0, 1.0, 0.2, -1.0],
            ],
        ])

        self.xyzuvw_cov = np.array([
            [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                ],
                [
                    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
                ],
            ],
            [
                [
                    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
                ],
                [
                    [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 4.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
                ],
                [
                    [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
                ],
            ],
        ])

        self.star_pars = {
            'times': self.times, 'xyzuvw': self.xyzuvw,
            'xyzuvw_cov': self.xyzuvw_cov,
        }

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


    def test_interp_cov(self):
        """Test the interpolation between time steps"""
        target_time = 0.0
        interp_covs, interp_mns = gf.interp_cov(target_time, self.star_pars)
        self.assertTrue(np.allclose(interp_mns, self.xyzuvw[:, 0, :]))
        self.assertTrue(np.allclose(interp_covs, self.xyzuvw_cov[:, 0]))

        target_time = 1.0
        interp_covs, interp_mns = gf.interp_cov(target_time, self.star_pars)
        self.assertTrue(np.allclose(interp_mns, self.xyzuvw[:, 1, :]))
        self.assertTrue(np.allclose(interp_covs, self.xyzuvw_cov[:, 1]))

        target_time = 0.5
        interp_covs, interp_mns = gf.interp_cov(target_time, self.star_pars)
        # check covariance matrices have 1.5 and 3.5 along diagonals
        self.assertTrue(np.allclose(interp_covs[0], np.eye(6) * 1.5))
        self.assertTrue(np.allclose(interp_covs[1], np.eye(6) * 3.5))

        # first star mean should be average between timesteps
        self.assertTrue(
            np.allclose(
                (self.xyzuvw_cov[0, 0] + self.xyzuvw_cov[0, 1]) / 2.0,
                interp_covs[0]
            )
        )

        # second star should show no interpolation in mean
        self.assertTrue(np.allclose(self.xyzuvw[1, 0], interp_mns[1]))

    def find_nearest(self, array, value):
        """Basic tool to find the array element that is closest to a value"""
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def test_eig_prior(self):
        """Test that the eig_prior is maximal at the characteristic minima"""
        eig_vals = np.linspace(0.1, 20, 200)
        char_mins = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

        for ctr, char_min in enumerate(char_mins):
            self.assertEqual(
                eig_vals[np.argmax(gf.eig_prior(char_min, eig_vals))],
                self.find_nearest(eig_vals, char_min),
                msg="{}...".format(ctr),
            )

    def test_lnprior(self):
        """Test correctness of lnprior - only the final set of pars is valid"""
        many_group_pars = np.array([
            # X, Y, Z, U,  V,  W,1/dX,1/dY,1/dZ,1/dV,Cxy,Cxz,Cyz,age,nstars
            [0, 0, 0, 0, 0, 0, 1 / 10, 1 / 10, 1 / 10, 1 / 5, .5, .2, .3, -20, 100],
            [20, 20, 20, 5, 1e9, 5, 1 / 10, 1 / 10, 1 / 10, 1 / 5, -.3, -.6, .2, 1, 100],
            [50, 50, 50, 0, 0, -10, 1 / 10, 1 / 10, 1 / 10, 1 / 5, 1.8, .3, .2, 1, 100],
            [50, 50, 50, 0, 0, -10, 1 / 10, 1 / -5, 1 / 10, 1 / 5, -.8, .3, .2, 1, 100],
            [50, 50, 50, 0, 0, -10, 1 / 10, 1 / 10, 1 / 10, 1 / 5, -.8, .3, .2, 10, 100],
            [0, 0, 0, 0, 0, 0, 1 / 10, 1 / 10, 1 / 10, 1 / 5, .5, .2, .3, 1, 100],
        ])

        priors = np.zeros(many_group_pars.shape[0])

        for i in range(many_group_pars.shape[0]):
            priors[i] = gf.lnprior(many_group_pars[i], None, self.star_pars)

        self.assertEqual(np.max(priors[:-2]), -np.inf)
        self.assertEqual(priors[-1], 0.0)

    def test_lnlike_lnprob(self):
        """
        Compare likelihood results for the groups and given stars
        """
        # at t = 0.0 both stars are at around
        # [0.0,0.5,1.0,1.0,0.2,-1.0],
        # star 1 evolves but star 2 remains fixed

        # self.xyzuvw = np.array([
        #    [
        #        [ 0.0,0.5,1.0,1.1,0.2,-1.0],
        #        [-1.0,0.3,2.0,0.9,0.2,-1.0],
        #        [-1.8,0.1,3.0,0.7,0.2,-1.0],
        #    ],
        #    [
        #        [0.0,0.5,1.0,1.0,0.2,-1.0],
        #        [0.0,0.5,1.0,1.0,0.2,-1.0],
        #        [0.0,0.5,1.0,1.0,0.2,-1.0],

        z = np.ones(self.xyzuvw.shape[0])

        # identical as s1 at t=0 and 2 at all t's
        group_pars1 = np.array(
            [0.0, 0.5, 1.0, 1.0, 0.2, -1.0, 1 / 5, 1 / 5, 1 / 5, 1 / 2, 0, 0, 0, 0.0]
        )
        # same as s1 at t=0 but double the width
        group_pars2 = np.array(
            [0.0, 0.5, 1.0, 1.0, 0.2, -1.0, 1 / 10, 1 / 10, 1 / 10, 1 / 4, 0, 0, 0, 0.0]
        )
        # age is at 0.5, so s1 should have evolved away, but s2 is still
        # a good fit
        group_pars3 = np.array(
            [0.0, 0.5, 1.0, 1.0, 0.2, -1.0, 1 / 5, 1 / 5, 1 / 5, 1 / 2, 0, 0, 0, 0.5]
        )
        # way off from everything
        group_pars4 = np.array(
            [100, 100, 100, 50, 50, 50, 1 / 10, 1 / 10, 1 / 10, 1 / 4, 0, 0, 0, 0.0]
        )

        # assert lnlike: 1 > [2,3,4], [2,3] > 4
        self.assertTrue(
            gf.lnlike(group_pars1, z, self.star_pars) >
            gf.lnlike(group_pars2, z, self.star_pars)
        )
        self.assertTrue(
            gf.lnlike(group_pars1, z, self.star_pars) >
            gf.lnlike(group_pars3, z, self.star_pars)
        )
        self.assertTrue(
            gf.lnlike(group_pars1, z, self.star_pars) >
            gf.lnlike(group_pars4, z, self.star_pars)
        )
        self.assertTrue(
            gf.lnlike(group_pars2, z, self.star_pars) >
            gf.lnlike(group_pars4, z, self.star_pars)
        )
        self.assertTrue(
            gf.lnlike(group_pars3, z, self.star_pars) >
            gf.lnlike(group_pars4, z, self.star_pars)
        )

        # assert lnprob: 1 > [2,3,4], [2,3] > 4
        self.assertTrue(
            gf.lnprobfunc(group_pars1, z, self.star_pars) >
            gf.lnprobfunc(group_pars2, z, self.star_pars)
        )
        self.assertTrue(
            gf.lnprobfunc(group_pars1, z, self.star_pars) >
            gf.lnprobfunc(group_pars3, z, self.star_pars)
        )
        self.assertTrue(
            gf.lnprobfunc(group_pars1, z, self.star_pars) >
            gf.lnprobfunc(group_pars4, z, self.star_pars)
        )
        self.assertTrue(
            gf.lnprobfunc(group_pars2, z, self.star_pars) >
            gf.lnprobfunc(group_pars4, z, self.star_pars)
        )
        self.assertTrue(
            gf.lnprobfunc(group_pars3, z, self.star_pars) >
            gf.lnprobfunc(group_pars4, z, self.star_pars)
        )
        return 0

    def test_fit_group(self):
        """
        Synthesise a tb file with negligible error, retrieve initial
        parameters
        """

        # an 'external' parametrisation, with everything in physical
        # format
        group_pars_ex = np.array(
            # X, Y, Z, U,  V,  W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars
            [0, 0, 0, 0, 0, 0, 10, 10, 10, 5, -.3, -.6, .2, 10, 100],
        )

        # an 'internal' parametrisation, stds are listed as the inverse,
        # no need to know how many stars etc.
        group_pars_in = np.copy(group_pars_ex[:-1])
        group_pars_in[6:10] = 1 / group_pars_in[6:10]

        # neligible error, anything smaller runs into problems with matrix
        # inversions
        error = 1e-5
        ntimes = 20

        tb_file = "tmp_groupfitter_tb_file.pkl"

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
                1, group_pars_ex, error, savefile=self.synth_file
            )
            with open(self.synth_file, 'r') as fp:
                t = pickle.load(fp)

            times = np.linspace(0, 2 * group_pars_ex[-2], ntimes)
            tb.traceback(t, times, savefile=tb_file)

        # find best fit
        best_fit, _, _ = gf.fit_group(
            tb_file, burnin_steps=1000, sampling_steps=1000, plot_it=True
        )

        # this code belongs in expect_max
        #        # check membership list totals to nstars in group
        #        self.assertEqual(int(round(np.sum(memb))), group_pars[-1])
        #        self.assertEqual(round(np.max(memb)), 1.0)
        #        self.assertEqual(round(np.min(memb)), 1.0)

        ctr = 0  # left here for convenience... tidy up later

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


    def test_get_bayes_spreads(self):
        """Using a precise synthetic traceback, compare bayes and naive spread
        """
        mock_twa_pars = [
            -80, 80, 50, 10, -20, -5, 5, 5, 5, 2, 0.0, 0.0, 0.0, 7, 40
        ]
        error_perc = 1e-5
        ngroups = 1
        syn.synthesise_data(
            ngroups, mock_twa_pars, error_perc, savefile=self.synth_file
        )
        with open(self.synth_file, 'r') as fp:
            t = pickle.load(fp)

        ntimes = 6
        times = np.linspace(0, 10, ntimes)
        tb.traceback(t, times, savefile=self.tb_file)
        star_pars = gf.read_stars(self.tb_file)
        xyzuvw = star_pars['xyzuvw']
        bayes_spreads, time_probs = gf.get_bayes_spreads(self.tb_file)
        naive_spreads = an.get_naive_spreads(xyzuvw)

        self.assertTrue(np.isclose(bayes_spreads, naive_spreads, rtol=0.1).all())

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(GroupfitterTestCase)
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
