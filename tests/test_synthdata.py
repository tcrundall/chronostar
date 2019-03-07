"""
Tests the SynthData class which is used to generate synthetic stellar
kinematic data from simple, Gaussian distributions.
"""
from __future__ import print_function, division, unicode_literals

from astropy.table import Table, join
import numpy as np

import sys
sys.path.insert(0,'..')
from chronostar.synthdata import SynthData
from chronostar.component import SphereComponent, EllipComponent

PARS =  np.array([
    [0., 0., 0., 0., 0., 0., 10., 5., 1e-5],
    [5., 0.,-5., 0., 0., 0., 10., 5., 40.]
])
STARCOUNTS = [50, 30]
COMPONENTS = SphereComponent

def test_initialisation():
    """Basic sanity check to see if things start off ok"""
    sd = SynthData(pars=PARS, starcounts=STARCOUNTS, Components=COMPONENTS)

    assert np.allclose(PARS, sd.pars)
    assert sd.ncomps == len(PARS)
    assert np.allclose(PARS[0], sd.components[0].get_pars())
    assert np.allclose(np.array(STARCOUNTS), sd.starcounts)


    sd2 = SynthData(pars=PARS[0], starcounts=STARCOUNTS[0],
                    Components=COMPONENTS)
    assert np.allclose(np.array([STARCOUNTS[0]]), sd2.starcounts)

    starcounts = 50.
    sd3 = SynthData(pars=PARS[0], starcounts=starcounts,
                    Components=COMPONENTS)
    assert np.allclose(np.array([np.int(starcounts)]), sd3.starcounts)


def test_generateInitXYZUVW():
    """Check that the mean of initial xyzuvw of stars matches that of the
    initialising component"""
    starcounts = (int(1e6),)
    sd = SynthData(pars=PARS[:1], starcounts=starcounts, Components=COMPONENTS)
    sd.generateAllInitXYZUVW()

    comp = SphereComponent(PARS[0])
    init_xyzuvw = sd.extractDataAsArray([dim+'0' for dim in 'xyzuvw'])
    assert np.allclose(comp.get_mean(), np.mean(init_xyzuvw, axis=0),
                       atol=0.1)


def test_projectStars():
    """Check that the mean of stars after projection matches the mean
    of the component after projection"""
    starcounts = (int(1e3),)
    sd = SynthData(pars=PARS[:1], starcounts=starcounts, Components=COMPONENTS)
    sd.generateAllInitXYZUVW()
    sd.projectStars()

    comp_mean_now, comp_covmatrix_now = \
        sd.components[0].get_currentday_projection()
    final_xyzuvw = sd.extractDataAsArray([dim+'_now' for dim in 'xzyuvw'])
    assert np.allclose(comp_mean_now, final_xyzuvw.mean(axis=0), atol=1.)


def test_measureXYZUVW():
    """Check measurements of xyzuvw_now to astrometry occur properly.
    Will use extremely dense component as case study as this ensures stars
    all have more or less the same true values"""
    compact_comp_pars = np.copy(PARS[0])
    compact_comp_pars[6] = 1e-15
    compact_comp_pars[7] = 1e-15
    compact_comp_pars[8] = 1e-15
    starcounts = [1000]

    sd = SynthData(pars=np.array([compact_comp_pars]), starcounts=starcounts,
                   Components=COMPONENTS)
    sd.generateAllInitXYZUVW()
    sd.projectStars()
    sd.measureXYZUVW()

    for colname in SynthData.DEFAULT_ASTR_COLNAMES:
        assert np.allclose(sd.GERROR[colname + '_error'],
                           sd.astr_table[colname + '_error'])
        # Check spread of data is similar to Gaia error, we use
        # a large tolerance so a small number of stars can be used
        assert np.isclose(sd.GERROR[colname + '_error'],
                          np.std(sd.astr_table[colname]),
                          rtol=1e-1)


def test_storeTable():
    """Check storing table and loading works"""
    filename = 'temp_data/test_storeTable_output.fits'
    sd = SynthData(pars=PARS, starcounts=STARCOUNTS, Components=COMPONENTS)
    sd.synthesiseEverything()
    sd.storeTable(filename=filename, overwrite=True)
    stored_table = Table.read(filename)

    assert np.allclose(sd.astr_table['parallax'], stored_table['parallax'])


def test_synthesiseEverything():
    """Check everything goes to plan with single call"""
    sd = SynthData(pars=PARS, starcounts=STARCOUNTS, Components=COMPONENTS)
    sd.synthesiseEverything()

    assert np.isclose(np.sum(STARCOUNTS), len(sd.astr_table))


def test_storeAndLoad():
    """Check that storing and loading works as expected"""
    filename = 'temp_data/test_synthesiseEverything_output.fits'
    sd = SynthData(pars=PARS, starcounts=STARCOUNTS, Components=COMPONENTS)
    sd.synthesiseEverything(filename=filename, overwrite=True)

    # Trying to store table at `filename` without overwrite throws error
    try:
        sd.synthesiseEverything(filename=filename, overwrite=False)
    except IOError:
        pass

    #TODO: implement some means of storing (and loading) entire object


def test_artificialMeasurement():
    """Ensure that scaling the measurement uncertainty scales the reported
    uncertainties appropriately, and that offsets in data due to error scale
    with input error"""
    pars = PARS[:1]
    starcounts = [100]
    sd_dict = {}
    names = ['perf', 'good', 'norm', 'bad']
    m_err_dict = {
        'perf':1e-10,
        'good':1e-1,
        'norm':1.0,
        'bad':1e1,
    }
    for name in names:
        np.random.seed(1)
        sd = SynthData(pars=pars, starcounts=starcounts,
                       measurement_error=m_err_dict[name],
                       Components=COMPONENTS)
        sd.synthesiseEverything()
        sd_dict[name] = sd

    # Assert that measurement errors are stored correctly in columns
    for name in names[1:]:
        assert np.allclose(
                sd_dict[name].astr_table['radial_velocity_error'],
                m_err_dict[name]*SynthData.GERROR['radial_velocity_error']
        )

    # Get reference for degree of offset expected
    norm_offset = np.mean(
            np.abs(sd_dict['perf'].astr_table['radial_velocity']
                   - sd_dict['norm'].astr_table['radial_velocity'])
    )

    bad_offset = np.mean(
            np.abs(sd_dict['perf'].astr_table['radial_velocity']
                   - sd_dict['bad'].astr_table['radial_velocity'])
    )
    good_offset = np.mean(
            np.abs(sd_dict['perf'].astr_table['radial_velocity']
                   - sd_dict['good'].astr_table['radial_velocity'])
    )

    # Check the average offset scales with incorporated measurement error
    assert np.isclose(norm_offset*m_err_dict['bad'], bad_offset)
    assert np.isclose(norm_offset*m_err_dict['good'], good_offset)

if __name__ == '__main__':
    pass
