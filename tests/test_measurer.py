"""
Test the measurer module.

Checks the incorporated measurement uncertainties behave as expected, which
is normally distributed about the mean (true) with std of the defined
uncertainty
"""
import numpy as np
import sys
sys.path.insert(0, '..')
import chronostar.measurer as ms
import chronostar.coordinate as cc

def test_measureXYZUVW():
    """
    Measures beta pic `NMEASURES`, confirms spread in measurement
    is consistent with error
    """
    NMEASURES = 1000
    xyzuvw_bp_helio = np.array([-3.4, -16.4, -9.9, -11.0, -16.0, -9.1])
    xyzuvw = cc.convertHelioToLSR(xyzuvw_bp_helio)
    xyzuvws = np.tile(xyzuvw, (NMEASURES, 1))

    ref_errors = np.array([
        0., 0., ms.GERROR['e_Plx'], ms.GERROR['e_pm'],
        ms.GERROR['e_pm'], ms.GERROR['e_RV']
    ])
    astro_table = ms.measureXYZUVW(xyzuvws, 1.0)
    measured_vals, errors = ms.convertTableToArray(astro_table)
    assert np.allclose(ref_errors, np.std(measured_vals, axis=0), rtol=2e-1)