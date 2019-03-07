"""
Integration test.
Tests components of synthesiser, transform and traceorbit.
Tests the generation of synthetic associations based on sets of initial
distributions.

TODO: Actually write the tests
"""
import numpy as np
import sys
sys.path.insert(0, '..')
import chronostar.synthdata as syn

def testManyGroups():
    """
    The point of the function this unit_tests is to generate *AND PROJECT*
    a bunch of stars. The function does not project stars through time,
    because of the author's dislike of 'spaghetti' code, and the synthesiser
    module is desired to be separate from the traceorbit module.

    As a consequence this test is at the moment useless
    """
    if False:
        origins_pars = np.array([
           #  X     Y     Z    U    V    W   dX  dV  age   nstars
           [ 25.,   0.,  11., -5.,  0., -2., 10., 5.,  3.,    100.],
           [-21., -60.,   4.,  3., 10., -1.,  7., 3.,  7.,     80.],
           [-10.,  20.,   0.,  1., -4., 15., 10., 2., 10.,     90.],
           [-80.,  80., -80.,  5., -5.,  5., 20., 5., 13.,    120.],
        ])
        ngroups = origins_pars.shape[0]

        # construct the index boundaries
        ranges = [0]
        for i in range(ngroups):
            ranges.append(int(ranges[i] + origins_pars[i,-1]))

        init_xyzuvw, groups = syn.synthesiseManyXYZUVW(origins_pars, sphere=True,
                                                       return_groups=True)
        # check means of first lot of stars
        for i in range(ngroups):
            assert np.allclose(origins_pars[i,:6],
                               init_xyzuvw[ranges[i]:ranges[i+1]].mean(axis=0),
                               atol=5.)

        # check group objects initialised correctly
        for i in range(ngroups):
            assert np.allclose(origins_pars[i,:6],
                               groups[i].mean)
            assert np.allclose(origins_pars[i,6], groups[i].dx)
            assert np.allclose(origins_pars[i,7], groups[i].dv)
            assert np.allclose(origins_pars[i,8], groups[i].age)
            assert np.allclose(origins_pars[i,9], groups[i].nstars)


def testManyGroupsSave():
    """
    The point of the function this unit_tests is to generate *AND PROJECT*
    a bunch of stars. The function does not project stars through time,
    because of the author's dislike of 'spaghetti' code, and the synthesiser
    module is desired to be separate from the traceorbit module.

    As a consequence this test is at the moment useless
    """

    if False:
        groups_savefile = "temp_data/origins.npy"
        xyzuvw_savefile = "temp_data/all_init_xyzuvw.npy"
        origins = np.array([
           #  X    Y    Z    U    V    W   dX  dY    dZ  dVCxyCxzCyz age nstars
           [25., 0., 11., -5., 0., -2., 10., 5., 3., 100.],
           [-21., -60., 4., 3., 10., -1., 7., 3., 7., 80.],
           [-10., 20., 0., 1., -4., 15., 10., 2., 10., 90.],
           [-80., 80., -80., 5., -5., 5., 20., 5., 13., 120.],
        ])
        ngroups = origins.shape[0]

        # construct the index boundaries
        ranges = [0]
        for i in range(ngroups):
            ranges.append(int(ranges[i] + origins[i,-1]))

        syn.synthesiseManyXYZUVW(origins, sphere=True, return_groups=False,
                                 xyzuvw_savefile=xyzuvw_savefile,
                                 groups_savefile=groups_savefile)
        init_xyzuvw = np.load(xyzuvw_savefile)
        groups = np.load(groups_savefile)

        # check means of first lot of stars
        for i in range(ngroups):
            assert np.allclose(origins[i,:6],
                               init_xyzuvw[ranges[i]:ranges[i+1]].mean(axis=0),
                               atol=5.)
        # check group objects initialised correctly
        for i in range(ngroups):
            assert np.allclose(origins[i,:6],
                               groups[i].mean)
            assert np.allclose(origins[i,6], groups[i].dx)
            assert np.allclose(origins[i,7], groups[i].dv)
            assert np.allclose(origins[i,8], groups[i].age)
            assert np.allclose(origins[i,9], groups[i].nstars)