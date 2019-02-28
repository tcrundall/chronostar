import logging
import numpy as np
import sys

import chronostar.component

sys.path.insert(0, '..')

import chronostar.transform as tf
import chronostar.traceorbit as torb
import chronostar.synthdata as syn


def test_traceOrbitWithTransform():
    """Bit of a clunky interaction with allowing traceOrbitXYZUVW to take in
    either an array or a single value AND being able to pass that fucntion to
    another function with extra args passed as a tuple
    """
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    pars = np.array([0.,0.,0.,0.,0.,0.,0.,0.,1e-8])
    g = chronostar.component.Component(pars, internal=True)
    cov_then = g.generateCovMatrix()

    mean_now = torb.traceOrbitXYZUVW(g.mean, g.age, True)
    cov_now = tf.transformCovMat(
        cov_then, torb.traceOrbitXYZUVW, g.mean, dim=6, args=(g.age, True)
    )

    assert np.allclose(mean_now, g.mean)
    assert np.allclose(cov_now, cov_then, atol=1e-5)
