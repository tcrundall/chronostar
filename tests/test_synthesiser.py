import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.synthesiser as syn

def test_groupGeneration():
    pars = [0., 0., 0., 0., 0., 0., 10., 5., 20., 50.]
    myGroup = syn.SynthGroup(pars, sphere=True)
    assert myGroup.age == pars[-2]

    scmat = myGroup.generateSphericalCovMatrix()
    assert np.max(np.linalg.eigvalsh(scmat)) == np.max(pars[6:8])**2

def test_starGenerate():
    pars = [0., 0., 0., 0., 0., 0., 10., 5., 20., 50.]
    init_xyzuvw = syn.synthesise_xyzuvw(pars, sphere=True)
    myGroup = syn.SynthGroup(pars, sphere=True)
    np_cov = np.cov(init_xyzuvw.T)


