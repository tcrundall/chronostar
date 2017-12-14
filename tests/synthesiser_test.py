#!/usr/bin/env python -W ignore
"""
test_synthesiser
----------------------------

Tests for `synthesiser` module.

Takes about 10 mins to run on an old mac laptop

To Do:
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

class SynthesiserTestCase(unittest.TestCase):
    def setUp(self):
        self.many_group_pars = np.array([
            # X, Y, Z, U,  V,  W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars
            [ 0, 0, 0, 0,  0,  0,10,10,10, 5, .5, .2, .3, 20, 100],
            [20,20,20, 5,  5,  5,10,10,10, 5,-.3,-.6, .2, 20, 100],
            [50,50,50, 0,  0,-10,10,10,10, 5,-.8, .3, .2, 40, 100],
        ])
        self.tempdir = tempfile.mkdtemp()
        self.synth_file = os.path.join(self.tempdir, 'synth_data.pkl')
        self.tb_file    = os.path.join(self.tempdir, 'tb_data.pkl') 

    def tearDown(self):
        try:
            os.remove(self.synth_file)
        except OSError:
            pass
        try:
            os.remove(self.tb_file)
        except OSError:
            pass
        os.rmdir(self.tempdir)

    def test_stationary(self):
        """check stars are initialised in correct position"""
        for ctr, orig_group_pars in enumerate(self.many_group_pars):
            # copy group pars and set age to a negligbly small amount
            group_pars = np.copy(orig_group_pars) 
            group_pars[-2] = 0.01
            self.assertTrue(group_pars[-2] != 0)
            # increase number of stars for better statistical results
            group_pars[-1] = 1000
            xyzuvw_now = syn.synth_group(group_pars)
            
            threshold = 1.0
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

            tol = 0.1
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

    def test_exact(self):
        """
        perform a measurement with practically no error,
        ensure mean and std of XYZUVW are similar-ish
        """
        for ctr, group_pars in enumerate(self.many_group_pars):
            #print("... {}".format(ctr))
            #group_pars=[40,40,40, 0, -5, -5,10,10,10, 5, .0, .0, .0, 30, 100]
            self.assertTrue(group_pars[-2] != 0)

            # synthesise group data with negligible error
            # errors smaller than 1e-5 create problems with matrix inversions
            error = 1e-5 #0.00001               # 0.001% of gaia-esque error
            syn.synthesise_data(1, group_pars, error, savefile=self.synth_file)

            fp = open(self.synth_file, 'r')
            t = pickle.load(fp)
            fp.close()

            times = np.array([0, group_pars[-2]])

            #find stars in their 'original' conditions as traced back, see
            # if corresponds appropriately to intial group conditions
            tb.traceback(t,times,savefile=self.tb_file)
            
            fp = open(self.tb_file, 'r')
            stars,times,xyzuvw,xyzuvw_cov = pickle.load(fp)
            fp.close()

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

    def test_errors(self):
        """Have some test to check implemented errors are appropriate"""

        # initialise a group of negligible age with negliglbe spread in both
        # position and velocity. Hence, any dispersion in the generated
        # data table will be from introduced uncertainty

        #             X,Y,Z,U,V,W,  dX,  dY,  dZ,  dV,Cxy,Cxz,Cyz, age,nstars
        group_pars=[100,0,0,0,5,0,.001,.001,.001,.001, .0, .0, .0,.001, 500]
        
        errs = [0.5, 1.0, 2.0]  
        for err in errs:
            # Synthesise group, 
            syn.synthesise_data(1, group_pars, err, savefile=self.synth_file)
            t = pickle.load(open(self.synth_file, 'r'))

            Plx = t['Plx']
            e_Plx = t['e_Plx']
            RV = t['RV']
            e_RV  = t['e_RV']
            pmRA = t['pmRA']
            e_pmRA = t['e_pmRA']
            pmDE = t['pmDE']
            e_pmDE = t['e_pmDE']

            # for each measurement, confirm the the standard deviation of
            # all the stars are within some tolerance of the appropriately
            # scaled gaia-esque error

            tol = 0.10 #set tolerance within 10%; 100 stars achieves 20% tol
            self.assertTrue(
                abs(np.std(Plx) - err*syn.GAIA_ERRS['e_Plx']) <= \
                tol*err*syn.GAIA_ERRS['e_Plx'],
                msg="std Plx, factor {}, received {}, expected {}".\
                format(err, np.std(Plx), err*syn.GAIA_ERRS['e_Plx']))

            self.assertTrue(
                abs(np.mean(e_Plx) - err*syn.GAIA_ERRS['e_Plx']) <= \
                tol*err*syn.GAIA_ERRS['e_Plx'],
                msg="mn e_Plx, factor {}, received {}, expected {}".\
                format(err, np.mean(e_Plx), err*syn.GAIA_ERRS['e_Plx']))

            self.assertTrue(
                abs(np.std(RV) - err*syn.GAIA_ERRS['e_RV']) <= \
                tol*err*syn.GAIA_ERRS['e_RV'],
                msg="std RV, factor {}, received {}, expected {}".\
                format(err, np.std(RV), err*syn.GAIA_ERRS['e_RV']))

            self.assertTrue(
                abs(np.mean(e_RV) - err*syn.GAIA_ERRS['e_RV']) <= \
                tol*err*syn.GAIA_ERRS['e_RV'],
                msg="mn e_RV, factor {}, received {}, expected {}".\
                format(err, np.mean(e_RV), err*syn.GAIA_ERRS['e_RV']))

            self.assertTrue(
                abs(np.std(pmRA) - err*syn.GAIA_ERRS['e_pm']) <= \
                tol*err*syn.GAIA_ERRS['e_pm'],
                msg="std pmRA, factor {}, received {}, expected {}".\
                format(err, np.std(pmRA), err*syn.GAIA_ERRS['e_pm']))

            self.assertTrue(
                abs(np.mean(e_pmRA) - err*syn.GAIA_ERRS['e_pm']) <= \
                tol*err*syn.GAIA_ERRS['e_pm'],
                msg="mn e_pmRA, factor {}, received {}, expected {}".\
                format(err, np.mean(e_pmRA), err*syn.GAIA_ERRS['e_pm']))

            self.assertTrue(
                abs(np.std(pmDE) - err*syn.GAIA_ERRS['e_pm']) <= \
                tol*err*syn.GAIA_ERRS['e_pm'],
                msg="std pmDE, factor {}, received {}, expected {}".\
                format(err, np.std(pmDE), err*syn.GAIA_ERRS['e_pm']))

            self.assertTrue(
                abs(np.mean(e_pmDE) - err*syn.GAIA_ERRS['e_pm']) <= \
                tol*err*syn.GAIA_ERRS['e_pm'],
                msg="mn e_pmDE, factor {}, received {}, expected {}".\
                format(err, np.mean(e_pmDE), err*syn.GAIA_ERRS['e_pm']))

def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(SynthesiserTestCase)
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())

sys.path.insert(0,'.') # reinserting home directory into path

