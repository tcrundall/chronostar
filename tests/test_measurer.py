import numpy as np
import sys
sys.path.insert(0, '..')
import chronostar.measurer as ms
import chronostar.coordinate as cc

def test_measureXYZUVW():
    """
    Measures beta pic NSTARS times, confirms spread in measurement
    is consistent with error
    """
    NSTARS = 1000
    xyzuvw_bp_helio = np.array([-3.4, -16.4, -9.9, -11.0, -16.0, -9.1])
    xyzuvw = cc.convertHelioToLSR(xyzuvw_bp_helio)
    xyzuvws = np.tile(xyzuvw, (NSTARS, 1))

    ref_errors = np.array([
        0., 0., ms.GERROR['e_Plx'], ms.GERROR['e_pm'],
        ms.GERROR['e_pm'], ms.GERROR['e_RV']
    ])
    astro_table = ms.measureXYZUVW(xyzuvws, 1.0)
    measured_vals, errors = ms.convertTableToArray(astro_table)
    import pdb; pdb.set_trace()
    assert np.allclose(ref_errors, np.std(measured_vals, axis=0), rtol=2e-1)