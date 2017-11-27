#!/usr/bin/env python
"""
test_synthesiser
----------------------------

Tests for `synthesiser` module.
"""

import os.path
import pdb
import sys
import tempfile
import unittest

sys.path.insert(0,'..') #hacky way to get access to module

import chronostar.synthesiser as syn
import numpy as np
import pickle 
#from chronostar.fit_group import compute_overlap as co
#from chronostar._overlap import get_overlap as swig_co
#from chronostar._overlap import get_overlaps as swig_cos


class TestStringMethods(unittest.TestCase):
    def setUp(self):
        group_pars = np.array([
            # X, Y, Z, U,  V,  W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars
            [ 0, 0, 0, 0,  0,  0,10,10,10, 5, .0, .0, .0, 10, 100],
            [ 0, 0, 0, 0,  0,  0,10,10,10, 5, .0, .0, .0, 20, 100],
            [20,20,20, 5,  5,  5,10,10,10, 5, .0, .0, .0, 20, 100],
            [40,40,40, 0, -5, -5,10,10,10, 5, .0, .0, .0, 30, 100],
            [50,50,50, 0,  0,-10,10,10,10, 5, .0, .0, .0, 40, 100],
        ])
        self.tempdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tempdir, 'group_pars.pkl')
        with open(self.path, 'w') as f:
            pickle.dump(group_pars, f)

    #def tearDown(self):
    #    self.tempdir.cleanup()

    def test_synth(self):
        group_pars = pickle.load(open(self.path, 'r'))
        self.assertTrue(True)
if __name__ == '__main__':
    unittest.main()

sys.path.insert(0,'.') #hacky way to get access to module
