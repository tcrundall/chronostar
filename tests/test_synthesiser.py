#!/usr/bin/env python
"""
test_synthesiser
----------------------------

Tests for `synthesiser` module.
"""

import os.path
import sys
import tempfile
import unittest

sys.path.insert(0,'..') #hacky way to get access to module

import chronostar.synthesiser as syn
import chronostar.traceback as tb
import numpy as np
import pdb
import pickle 

#class TestStringMethods(unittest.TestCase):
class TestSynthesiser(unittest.TestCase):
    def setUp(self):
        self.many_group_pars = np.array([
            # X, Y, Z, U,  V,  W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars
            [ 0, 0, 0, 0,  0,  0,10,10,10, 5, .0, .0, .0, 10, 100],
            [ 0, 0, 0, 0,  0,  0,10,10,10, 5, .5, .2, .3, 20, 100],
            [20,20,20, 5,  5,  5,10,10,10, 5,-.3,-.6, .2, 20, 100],
            [40,40,40, 0, -5, -5,10,10,10, 5, .1, .2,-.3, 30, 100],
            [50,50,50, 0,  0,-10,10,10,10, 5,-.8, .3, .2, 40, 100],
        ])
        self.tempdir = tempfile.mkdtemp()
        self.synth_file = os.path.join(self.tempdir, 'synth_data.pkl')
        self.tb_file    = os.path.join(self.tempdir, 'tb_data.pkl') 

    #def tearDown(self):
    #    self.tempdir.cleanup()

    def test_stationary(self):
        """check stars are initialised in correct position"""
        for ctr, orig_group_pars in enumerate(self.many_group_pars):
            # copy group pars and set age to a negligbly small amount
            group_pars = np.copy(orig_group_pars) 
            group_pars[-2] = 0.01
            self.assertTrue(group_pars[-2] != 0)
            xyzuvw_now = syn.synth_group(group_pars)
            
            threshold = 3.5
            # check the mean values are approx what was initialised
            mean = np.mean(xyzuvw_now, axis=0)
            self.assertTrue(
                np.max(abs(mean - group_pars[0:6])) < threshold,
                msg="\nFailed {} received:\n{}\nshould be within {} to:\n{}".\
                format(ctr, mean, threshold, group_pars[0:6]))

            # check the standard dev values are approx what was initialised
            std = np.std(xyzuvw_now, axis=0)
            self.assertTrue(
                np.max(abs(std[:4] - group_pars[6:10])) < threshold,
                msg="\nFailed {} received:\n{}\nshould be within {} to:\n{}".\
                format(ctr, std[:4], threshold, group_pars[6:10]))
            
            # check correlation coefficients are within some tolerance
            # note, that don't need relative tolerance since -1<C<1
            Cxy = np.corrcoef(xyzuvw_now[:,0], xyzuvw_now[:,1])[0,1]
            Cxz = np.corrcoef(xyzuvw_now[:,0], xyzuvw_now[:,2])[0,1]
            Cyz = np.corrcoef(xyzuvw_now[:,1], xyzuvw_now[:,2])[0,1]
            Cuv = np.corrcoef(xyzuvw_now[:,3], xyzuvw_now[:,4])[0,1]
            tol = 0.25
            v_tol = 0.35

            self.assertTrue(abs(Cxy-group_pars[10]) <= tol,
                msg="\nFailed {} received:\n{}\nshould be within {} to:\n{}".\
                format(ctr, Cxy, tol, group_pars[10]))
            self.assertTrue(abs(Cxz-group_pars[11]) <= tol,
                msg="\nFailed {} received:\n{}\nshould be within {} to:\n{}".\
                format(ctr, Cxz, tol, group_pars[11]))
            self.assertTrue(abs(Cyz-group_pars[12]) <= tol,
                msg="\nFailed {} received:\n{}\nshould be within {} to:\n{}".\
                format(ctr, Cyz, tol, group_pars[12]))

            self.assertTrue(abs(Cuv) <= tol,
                msg="\nFailed {} received:\n{}\nshould be within {} to:\n{}".\
                format(ctr, Cuv, tol, 0))

            #pdb.set_trace()

    def test_exact(self):
        """
        perform a measurement with practically no error,
        ensure mean and std of XYZUVW are similar-ish
        """
        for ctr, group_pars in enumerate(self.many_group_pars):
            #group_pars=[40,40,40, 0, -5, -5,10,10,10, 5, .0, .0, .0, 30, 100]
            self.assertTrue(group_pars[-2] != 0)

            # synthesise group data with negligible error
            error = 0.001               # 0.1% of gaia-esque error
            syn.synthesise_data(1, group_pars, error, savefile=self.synth_file)

            t = pickle.load(open(self.synth_file, 'r'))
            times = np.array([0, group_pars[-2]])

            #find stars in their 'original' conditions as traced back, see
            # if corresponds appropriately to intial group conditions
            tb.traceback(t,times,savefile=self.tb_file)
            stars,times,xyzuvw,xyzuvw_cov = pickle.load(open(self.tb_file,'r'))

            threshold = 3.5
            mean = np.mean(xyzuvw[:,-1], axis=0)
            std  = np.std( xyzuvw[:,-1], axis=0)

            self.assertTrue(
                np.max(abs(mean - group_pars[0:6])) < threshold,
                msg="\nFailed {} received:\n{}\nshould be within {} to:\n{}".\
                format(ctr, mean, threshold, group_pars[0:6]))
            self.assertTrue(
                np.max(abs(std[:4] - group_pars[6:10])) < threshold,
                msg="\nFailed {} received:\n{}\nshould be close to:\n{}".\
                format(ctr, std[:4], group_pars[6:10]))

            # check correlation coefficients are within some tolerance
            # note, that don't need relative tolerance since -1<C<1
            Cxy = np.corrcoef(xyzuvw[:,-1,0], xyzuvw[:,-1,1])[0,1]
            Cxz = np.corrcoef(xyzuvw[:,-1,0], xyzuvw[:,-1,2])[0,1]
            Cyz = np.corrcoef(xyzuvw[:,-1,1], xyzuvw[:,-1,2])[0,1]
            Cuv = np.corrcoef(xyzuvw[:,-1,3], xyzuvw[:,-1,4])[0,1]
            tol = 0.25
            v_tol = 0.35

            self.assertTrue(abs(Cxy-group_pars[10]) <= tol,
                msg="\nFailed {} received:\n{}\nshould be within {} to:\n{}".\
                format(ctr, Cxy, tol, group_pars[10]))
            self.assertTrue(abs(Cxz-group_pars[11]) <= tol,
                msg="\nFailed {} received:\n{}\nshould be within {} to:\n{}".\
                format(ctr, Cxz, tol, group_pars[11]))
            self.assertTrue(abs(Cyz-group_pars[12]) <= tol,
                msg="\nFailed {} received:\n{}\nshould be within {} to:\n{}".\
                format(ctr, Cyz, tol, group_pars[12]))

            self.assertTrue(abs(Cuv) <= v_tol,
                msg="\nFailed {} received:\n{}\nshould be within {} to:\n{}".\
                format(ctr, Cuv, v_tol, 0))


if __name__ == '__main__':
    unittest.main()

sys.path.insert(0,'.') #hacky way to get access to module
