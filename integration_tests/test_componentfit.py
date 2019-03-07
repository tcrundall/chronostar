"""
Run some simple-ish test cases to see if ComponentFit
works
"""

import numpy as np
import sys
sys.path.inset(0, '..')

from chronostar.component import Component
from chronostar.synthdata import SynthData
from chronostar.ignore_componentfit import ComponentFit

def test_zeroAge():
    synth_star_pars = np.array([

    ])