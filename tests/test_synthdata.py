"""
Tests the SynthData class which is used to generate synthetic stellar
kinematic data from simple, Gaussian distributions.
"""
from __future__ import print_function, division, unicode_literals

from astropy.table import Table, join
import astropy.table as t
import numpy as np

import sys
sys.path.insert(0,'..')
from chronostar.synthdata import SynthData
from chronostar.component import Component

PARS =  np.array([
    [0., 0., 0., 0., 0., 0., 10., 5., 1e-5],
    [5., 0.,-5., 0., 0., 0., 10., 5., 40.]
])
STARCOUNTS = [50, 30]
COMP_FORMS = 'sphere'

def test_initialisation():
    """Basic sanity check to see if things start off ok"""
    sd = SynthData(pars=PARS, starcounts=STARCOUNTS, comp_forms=COMP_FORMS)

    assert np.allclose(PARS, sd.pars)
    assert sd.ncomps == len(PARS)
    assert np.allclose(PARS[0], sd.components[0].pars)


def test_generateInitXYZUVW():
    """Check that the mean of initial xyzuvw of stars matches that of the
    initialising component"""
    starcounts = (int(1e6),)
    sd = SynthData(pars=PARS[:1], starcounts=starcounts, comp_forms=COMP_FORMS)
    sd.generateAllInitXYZUVW()

    comp = Component(PARS[0])
    init_xyzuvw = sd.extractDataAsArray([dim+'0' for dim in 'xyzuvw'])
    assert np.allclose(comp.mean, np.mean(init_xyzuvw, axis=0), atol=0.1)


def test_projectStars():
    """Check that the mean of stars after projection matches the mean
    of the component after projection"""
    starcounts = (int(1e3),)
    sd = SynthData(pars=PARS[:1], starcounts=starcounts, comp_forms=COMP_FORMS)
    sd.generateAllInitXYZUVW()
    sd.projectStars()

    comp_mean_now, comp_covmatrix_now = \
        sd.components[0].getCurrentDayProjection()
    final_xyzuvw = sd.extractDataAsArray([dim+'_now' for dim in 'xzyuvw'])
    assert np.allclose(comp_mean_now, final_xyzuvw.mean(axis=0), atol=1.)


def test_measureXYZUVW():
    """Check measurements of xyzuvw_now to astrometry occur properly.
    Will use extremely dense component as case study"""
    compact_comp_pars = np.copy(PARS[0])
    compact_comp_pars[6] = 1e-5
    compact_comp_pars[7] = 1e-5
    compact_comp_pars[8] = 1e-5
    starcounts = [1000]

    sd = SynthData(pars=np.array([compact_comp_pars]), starcounts=starcounts,
                   comp_forms=COMP_FORMS)
    sd.generateAllInitXYZUVW()
    sd.projectStars()
    sd.measureXYZUVW()

    for colname in SynthData.DEFAULT_ASTR_COLNAMES:
        assert np.allclose(sd.GERROR[colname + '_error'],
                           sd.astr_table[colname + '_error'])
        assert np.isclose(sd.GERROR[colname + '_error'],
                          np.std(sd.astr_table[colname]),
                      rtol=5e2)


def test_storeTable():
    filename = 'temp_data/test_storeTable_output.fits'
    sd = SynthData(pars=PARS, starcounts=STARCOUNTS, comp_forms=COMP_FORMS)
    sd.generateAllInitXYZUVW()
    sd.projectStars()
    sd.measureXYZUVW()
    sd.storeTable(filename=filename, overwrite=True)
    stored_table = Table.read(filename)

    assert np.allclose(sd.astr_table['parallax'], stored_table['parallax'])


def test_synthesiseEverything():
    """Check everything goes to plan with single call"""
    filename = 'temp_data/test_synthesiseEverything_output.fits'
    sd = SynthData(pars=PARS, starcounts=STARCOUNTS, comp_forms=COMP_FORMS)
    sd.synthesiseEverything(filename=filename)

    assert np.isclose(np.sum(STARCOUNTS), len(sd.astr_table))


if __name__ == '__main__':
    starcounts = (int(1e2), 10)
    sd = SynthData(pars=PARS, starcounts=starcounts, comp_forms=COMP_FORMS)
    sd.generateAllInitXYZUVW()
    sd.projectStars()
    sd.measureXYZUVW()
    sd.storeTable(savedir='temp_data', filename='dummy_synth_data.fits',
                  overwrite=True)

