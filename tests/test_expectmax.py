#!/usr/bin/env python -W ignore
"""
test_expectmax
-----------------------------

Tests for `expectmax` module
"""
from __future__ import division, print_function

import sys
import numpy as np

sys.path.insert(0, '..')  # hacky way to get access to module

import chronostar.expectmax as em
import chronostar.synthesiser as syn
import chronostar.measurer as ms
import chronostar.converter as cv

group_pars_ex = np.array([
    # X, Y, Z, U, V, W, dX, dV, age,nstars
    [ 0, 0, 0, 0, 0, 0, 10,  5,  10,  500],
    [50,50, 0, 0, 0, 0, 10,  5,  10,  500],
])

data_dir = 'temp_data/'
synth_file = data_dir + 'synth_data.pkl'
tb_file = data_dir + 'tb_data.pkl'


def test_calc_errors():
    means = np.array([10,50,100,10,10,10, .2, .2, .2,  .1,  0 , 0,  0,10])
    stds  = np.array([ 2, 5, 10, 2, 2, 2,.05,.05,.05,.025,.05,.05,.05, 1])


def test_maximisation():
    """
    Synthesise a tb file with negligible error, retrieve initial
    parameters
    """
    group_pars_ex = np.array([
        # X, Y, Z, U, V, W, dX, dV, age,nstars
        [ 0, 0, 0, 0, 0, 0, 10,  5,  10,  10],
        [50,50, 0, 0, 0, 0, 10,  5,  10,  10],
    ])
    ngroups = group_pars_ex.shape[0]
    nstars = np.sum(group_pars_ex[:,-1])
    z = np.zeros((nstars, ngroups))

    start = 0
    for i in range(ngroups):
        nstars_in_group = group_pars_ex[i,-1]
        z[start:start+nstars_in_group,i] = 1.0
        start += nstars_in_group

    init_xyzuvw, origins = syn.synthesiseManyXYZUVW(group_pars_ex,
                                                    return_groups=True,
                                                    )
    astro_table = ms.measureXYZUVW(init_xyzuvw, 0.1)
    star_pars = cv.convertMeasurementsToCartesian(astro_table)

    import pdb; pdb.set_trace()
    # find best fit
    best_fits, _, _ = em.maximise(
        star_pars, ngroups, z=z, burnin_steps=200, sampling_steps=200,
    )
    pdb.set_trace()

    for ctr, (best_fit, group_pars_ex) in\
            enumerate(zip(best_fits, group_pars_ex)):
        means = best_fit[0:6]
        dx = 1. / best_fit[6]
        dv = 1. / best_fit[7]
        age = best_fit[8]

        tol_mean = 3.5
        tol_std = 2.5
        tol_age = 0.5

        assert (
            np.max(abs(means - group_pars_ex[0:6])) < tol_mean,
            "\nFailed {} received:\n{}\nshould be within {} to:\n{}".
            format(ctr, means, tol_mean, group_pars_ex[0:6]))
        assert (
            np.max(abs(dx - group_pars_ex[6])) < tol_std,
            "\nFailed {} received:\n{}\nshould be close to:\n{}".
            format(ctr, dx, tol_std, group_pars_ex[6:10]))
        assert (
            np.max(abs(dv - group_pars_ex[7])) < tol_std,
            "\nFailed {} received:\n{}\nshould be close to:\n{}".
                format(ctr, dv, tol_std, group_pars_ex[6:10]))
        assert (
            np.max(abs(age - group_pars_ex[8])) < tol_age,
            "\nFailed {} received:\n{}\nshould be close to:\n{}". \
            format(ctr, age, tol_age, group_pars_ex[13]))

"""
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

"""
