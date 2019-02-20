import logging
import numpy as np
from scipy.stats import mstats
import sys
sys.path.insert(0, '..')

import chronostar.synthesiser as syn
LOGGINGLEVEL = logging.DEBUG

TEMP_SAVE_DIR = 'temp_data/'

def testGroupGeneration():
    """Basic sanity checks for generation of Group objects from parameter
    input
    """
    logging.basicConfig(level=LOGGINGLEVEL,
                        filename="temp_logs/test_synthesiser.log")
    pars = [0., 0., 0., 0., 0., 0., 10., 5., 20., 50.]
    myGroup = syn.Group(pars, sphere=True)
    assert myGroup.age == pars[-2]

    scmat = myGroup.generateSphericalCovMatrix()
    assert np.max(np.linalg.eigvalsh(scmat)) == np.max(pars[6:8])**2

def testStarGenerate():
    """Check that stars generated from parameter input are distributed
    correctly"""
    logging.basicConfig(level=LOGGINGLEVEL,
                        filename="temp_logs/test_synthesiser.log")
    pars = [0., 0., 0., 0., 0., 0., 10., 5., 20., 100000.]
    init_xyzuvw = syn.synthesiseXYZUVW(pars, sphere=True)
    myGroup = syn.Group(pars, sphere=True)
    fitted_cov = np.cov(init_xyzuvw.T)
    np.set_printoptions(suppress=True, precision=4)
    assert np.allclose(fitted_cov, myGroup.generateSphericalCovMatrix(),
                       atol=2),\
        "\n{}\ndoesn't equal\n{}".format(
            fitted_cov, myGroup.generateSphericalCovMatrix()
        )
    fitted_dx = np.sqrt(mstats.gmean((
        fitted_cov[0,0], fitted_cov[1,1], fitted_cov[2,2]
    )))
    assert np.isclose(pars[6], fitted_dx, rtol=1e-2)

def testInternalPars():
    log_dx = 0.
    log_dv = 0.
    internal_pars_sphere = np.array([0.,0.,0.,0.,0.,0.,log_dx,log_dv,0.])
    myGroup = syn.Group(internal_pars_sphere, internal=True)
    assert myGroup.dx == np.exp(log_dx)
    assert myGroup.dv == np.exp(log_dv)
    assert np.allclose(myGroup.generateSphericalCovMatrix(), np.eye(6,6))

def testEllipticalGeneration():
    logging.basicConfig(level=LOGGINGLEVEL,
                        filename="temp_logs/test_synthesiser.log")
    pars = [0., 0., 0., 0., 0., 0., 10., 15., 5., 5., 0.0, 0.0, 0.0, 20.,
            100000.]
    init_xyzuvw = syn.synthesiseXYZUVW(pars, sphere=False)
    myGroup = syn.Group(pars, sphere=False)
    sphere_dx = mstats.gmean(pars[6:9])
    assert sphere_dx == myGroup.sphere_dx

    fitted_cov = np.cov(init_xyzuvw.T)
    fitted_sphere_dx = mstats.gmean((
        fitted_cov[0,0], fitted_cov[1,1], fitted_cov[2,2]
    ))**0.5
    assert np.isclose(sphere_dx, fitted_sphere_dx, rtol=1e-2)

    assert np.allclose(fitted_cov, myGroup.generateEllipticalCovMatrix(),
                       atol=3), \
        "\n{}\ndoesn't equal\n{}".format(
            fitted_cov, myGroup.generateEllipticalCovMatrix()
        )


def testSaveFiles():
    logging.basicConfig(level=LOGGINGLEVEL,
                        filename="temp_logs/test_synthesiser.log")
    xyzuvw_savefile = TEMP_SAVE_DIR+'temp_xyzuvw.npy'
    group_savefile = TEMP_SAVE_DIR+'temp_group.npy'
    pars = [0., 0., 0., 0., 0., 0., 10., 5., 20., 100.]
    xyzuvw_init, group = syn.synthesiseXYZUVW(
        pars, return_group=True, xyzuvw_savefile=xyzuvw_savefile,
        group_savefile=group_savefile
    )
    xyzuvw_saved = np.load(xyzuvw_savefile)
    assert np.allclose(xyzuvw_init, xyzuvw_saved)

    group_saved = np.load(group_savefile).item()
    assert group == group_saved


def testManyGroups():
    """
    The point of the function this tests is to generate *AND PROJECT*
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
    The point of the function this tests is to generate *AND PROJECT*
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