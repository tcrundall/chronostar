import logging
import numpy as np
from scipy.stats import mstats
import sys

import chronostar.component

sys.path.insert(0, '..')

import chronostar.synthdata as syn
LOGGINGLEVEL = logging.DEBUG

TEMP_SAVE_DIR = 'temp_data/'

def testGroupGeneration():
    """Basic sanity checks for generation of Group objects from parameter
    input
    """
    logging.basicConfig(level=LOGGINGLEVEL,
                        filename="temp_logs/test_synthesiser.log")
    pars = [0., 0., 0., 0., 0., 0., 10., 5., 20., 50.]
    myGroup = chronostar.component.Component(pars, form='sphere')
    assert myGroup.age == pars[-2]

    scmat = myGroup.generateSphericalCovMatrix()
    assert np.max(np.linalg.eigvalsh(scmat)) == np.max(pars[6:8])**2

def testStarGenerate():
    """Check that stars generated from parameter input are distributed
    correctly"""
    logging.basicConfig(level=LOGGINGLEVEL,
                        filename="temp_logs/test_synthesiser.log")
    pars = [0., 0., 0., 0., 0., 0., 10., 5., 20., 100000.]
    init_xyzuvw = syn.synthesiseXYZUVW(pars, form='sphere')
    myGroup = chronostar.component.Component(pars, form='sphere')
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
    """
    Check the generation of Group object with "internal" format (i.e.
    dx and dv provided in log space) is correct
    """
    log_dx = 0.
    log_dv = 0.
    internal_pars_sphere = np.array([0.,0.,0.,0.,0.,0.,log_dx,log_dv,0.])
    myGroup = chronostar.component.Component(internal_pars_sphere,
                                             internal=True)
    assert myGroup.dx == np.exp(log_dx)
    assert myGroup.dv == np.exp(log_dv)
    assert np.allclose(myGroup.generateSphericalCovMatrix(), np.eye(6,6))

def testEllipticalGeneration():
    """
    Checks correctness of the generation of Group object with
    non-isotropic (uncorrelated) position ellipsoid
    """
    logging.basicConfig(level=LOGGINGLEVEL,
                        filename="temp_logs/test_synthesiser.log")
    pars = [0., 0., 0., 0., 0., 0., 10., 15., 5., 5., 0.0, 0.0, 0.0, 20.]
    nstars = 1000000
    init_xyzuvw = syn.synthesiseXYZUVW(pars, nstars=nstars, form='elliptical')
    myGroup = chronostar.component.Component(pars, form='elliptical')
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
    """
    Checks files are saved in a way that's readable.
    """
    logging.basicConfig(level=LOGGINGLEVEL,
                        filename="temp_logs/test_synthesiser.log")
    xyzuvw_savefile = TEMP_SAVE_DIR+'temp_xyzuvw.npy'
    group_savefile = TEMP_SAVE_DIR+'temp_group.npy'
    pars = [0., 0., 0., 0., 0., 0., 10., 5., 20., 100.]
    xyzuvw_init, group = syn.synthesiseXYZUVW(pars, return_group=True,
                                              xyzuvw_savefile=xyzuvw_savefile,
                                              group_savefile=group_savefile)
    xyzuvw_saved = np.load(xyzuvw_savefile)
    assert np.allclose(xyzuvw_init, xyzuvw_saved)

    group_saved = np.load(group_savefile).item()
    assert group == group_saved

