#!/usr/bin/env python -W ignore
"""
test_expectmax
-----------------------------

Tests for `expectmax` module
"""
from __future__ import division, print_function

import logging
import numpy as np
import sys

sys.path.insert(0, '..')  # hacky way to get access to module

import chronostar.expectmax as em
import chronostar.synthesiser as syn
import chronostar.measurer as ms
import chronostar.converter as cv

group_pars = np.array([
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
    logging.basicConfig(level=logging.INFO, filemode='w',
                        filename='temp_logs/test_maximisation.log')
    group_pars = np.array([
        # X, Y, Z, U, V, W, dX, dV, age,nstars
        [ 0, 0, 0, 0, 0, 0, 10.,  5,  10,  100],
        [50,50, 0, 0, 0, 0, 10.,  5,  10,  100],
    ])
    ngroups = group_pars.shape[0]
    nstars = int(np.sum(group_pars[:,-1]))
    z = np.zeros((nstars, ngroups))

    # initialise z appropriately
    start = 0
    for i in range(ngroups):
        nstars_in_group = int(group_pars[i,-1])
        z[start:start+nstars_in_group,i] = 1.0
        start += nstars_in_group

    # generate data
    init_xyzuvw, origins = syn.synthesiseManyXYZUVW(group_pars,
                                                    return_groups=True,
                                                    internal=False,
                                                    )
    astro_table = ms.measureXYZUVW(init_xyzuvw, 1.0)
    star_pars = cv.convertMeasurementsToCartesian(astro_table)

    import pdb;
    all_init_pars = [o.getInternalSphericalPars() for o in origins]

    # perform maximisation step
    best_groups, all_samples, all_lnprob, all_init_pos =\
        em.maximisation(
            star_pars, ngroups, z, burnin_steps=100, idir=data_dir,
            all_init_pars=all_init_pars, plot_it=True
        )
    pdb.set_trace()

    # compare fit with input
    for origin, best_group in zip(origins, best_groups):
        o_pars = origin.getSphericalPars()
        b_pars = best_group.getSphericalPars()

        logging.info("origin pars:   {}".format(o_pars))
        logging.info("best fit pars: {}".format(b_pars))
        assert np.allclose(origin.mean, best_group.mean, atol=5.)
        assert np.allclose(origin.sphere_dx, best_group.sphere_dx, atol=2.)
        assert np.allclose(origin.dv, best_group.dv, atol=2.)
        assert np.allclose(origin.age, best_group.age, atol=1.)
    pdb.set_trace()


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
