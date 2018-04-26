import logging
import numpy as np
from scipy.stats import mstats
import sys
sys.path.insert(0, '..')

import chronostar.synthesiser as syn
LOGGINGLEVEL = logging.DEBUG

def testGroupGeneration():
    logging.basicConfig(level=LOGGINGLEVEL, filename="test_synthesiser.log")
    pars = [0., 0., 0., 0., 0., 0., 10., 5., 20., 50.]
    myGroup = syn.SynthGroup(pars, sphere=True)
    assert myGroup.age == pars[-2]

    scmat = myGroup.generateSphericalCovMatrix()
    assert np.max(np.linalg.eigvalsh(scmat)) == np.max(pars[6:8])**2

def testStarGenerate():
    logging.basicConfig(level=LOGGINGLEVEL, filename="test_synthesiser.log")
    pars = [0., 0., 0., 0., 0., 0., 10., 5., 20., 100000.]
    init_xyzuvw = syn.synthesise_xyzuvw(pars, sphere=True)
    myGroup = syn.SynthGroup(pars, sphere=True)
    fitted_cov = np.cov(init_xyzuvw.T)
    fitted_dx = mstats.gmean((
        fitted_cov[0,0], fitted_cov[1,1], fitted_cov[2,2]
    ))
    np.set_printoptions(suppress=True, precision=4)
    assert np.allclose(fitted_cov, myGroup.generateSphericalCovMatrix(),
                       atol=2),\
        "\n{}\ndoesn't equal\n{}".format(
            fitted_cov, myGroup.generateSphericalCovMatrix()
        )

def testEllipticalGeneration():
    logging.basicConfig(level=LOGGINGLEVEL,
                        filename="logs/test_synthesiser.log")
    pars = [0., 0., 0., 0., 0., 0., 10., 15., 5., 5., 0.0, 0.0, 0.0, 20.,
            100000.]
    init_xyzuvw = syn.synthesise_xyzuvw(pars, sphere=False)
    myGroup = syn.SynthGroup(pars, sphere=False)
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
    logging.basicConfig(level=LOGGINGLEVEL, filename="test_synthesiser.log")
    xyzuvw_savefile = 'temp_xyzuvw.npy'
    group_savefile = 'temp_group.npy'
    pars = [0., 0., 0., 0., 0., 0., 10., 5., 20., 100.]
    xyzuvw_init, group = syn.synthesise_xyzuvw(
        pars, return_group=True, xyzuvw_savefile=xyzuvw_savefile,
        group_savefile=group_savefile
    )
    xyzuvw_saved = np.load(xyzuvw_savefile)
    assert np.allclose(xyzuvw_init, xyzuvw_saved)

    group_saved = np.load(group_savefile).item()
    assert group == group_saved

