#!/usr/bin/env python 
"""
test_groupfitter
-----------------------------

Tests for `groupfitter` module
"""
from __future__ import division, print_function

from emcee.utils import MPIPool
import os.path
import pdb
import sys
import tempfile
import unittest

sys.path.insert(0,'..') #hacky way to get access to module

#import chronostar.fit_group as fg
import chronostar.groupfitter as gf
import chronostar.synthesiser as syn
import chronostar.traceback as tb
#from sympy.utilities.iterables import multiset_permutations
import numpy as np
import pickle

class TestGroupfitter(unittest.TestCase):
    def setUp(self):
        self.group_pars = np.array(
            # X, Y, Z, U,  V,  W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars
            [ 0,50,20, 5,  5,  5,10,10,10, 5,-.3,-.6, .2, 20, 100],
        )
        self.times = np.array([0.0, 1.0, 2.0])

        self.xyzuvw = np.array([
            [
                [ 0.0,0.5,1.0,1.1,0.2,-1.0],
                [-1.0,0.3,2.0,0.9,0.2,-1.0],
                [-1.2,0.1,3.0,0.7,0.2,-1.0],
            ],
            [
                [0.0,0.5,1.0,1.0,0.2,-1.0],
                [0.0,0.5,1.0,1.0,0.2,-1.0],
                [0.0,0.5,1.0,1.0,0.2,-1.0],
            ],
        ])

        self.xyzuvw_cov = np.array([
            [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                ],
                [
                    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
                ],
            ],
            [
                [
                    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
                ],
                [
                    [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 4.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
                ],
                [
                    [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
                ],
            ],
        ])

        self.star_pars = {
            'times':self.times,'xyzuvw':self.xyzuvw,
            'xyzuvw_cov':self.xyzuvw_cov,
        }

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

    def test_generate_icov(self):
        dX = 5
        dY = 5
        dZ = 5
        dV = 3
        Cxy = 0.0
        Cxz = 0.0
        Cyz = 0.0

        #                X,Y,Z,U,V,W,1/dX,1/dY,1/dZ,1/dV,Cxy,Cxz,Cyz,age
        pars = np.array([0,0,0,0,0,0,1/dX,1/dY,1/dZ,1/dV,Cxy,Cxz,Cyz,10])

        test_cov = np.array([
            [dX**2,     Cxy*dX*dY, Cxz*dX*dZ, 0.0,   0.0,   0.0],
            [Cxy*dX*dY, dY**2,     Cyz*dY*dZ, 0.0,   0.0,   0.0],
            [Cxz*dX*dZ, Cyz*dY*dZ, dZ**2,     0.0,   0.0,   0.0],
            [0.0,       0.0,       0.0,       dV**2, 0.0,   0.0],
            [0.0,       0.0,       0.0,       0.0,   dV**2, 0.0],
            [0.0,       0.0,       0.0,       0.0,   0.0,   dV**2],
        ])

        gf_cov = np.linalg.inv(gf.generate_icov(pars))
        
        self.assertTrue(np.allclose(test_cov, test_cov))
        self.assertTrue(np.allclose(test_cov, gf_cov))

        # orig calculation
        orig_cov = np.eye( 6 )
        #Fill in correlations
        orig_cov[np.tril_indices(3,-1)] = pars[10:13]
        orig_cov[np.triu_indices(3,1)] = pars[10:13]
        #Note that 'pars' stores the inverse of the standard deviation
        #Convert correlation to orig_covariance for position.
        for i in range(3):
            orig_cov[i,:3] *= 1 / pars[6:9]
            orig_cov[:3,i] *= 1 / pars[6:9]
        #Convert correlation to orig_covariance for velocity.
        for i in range(3,6):
            orig_cov[i,3:] *= 1 / pars[9]
            orig_cov[3:,i] *= 1 / pars[9]
        #Generate inverse cov matrix and its determinant
        self.assertTrue(np.allclose(orig_cov, gf_cov))
    
    def test_interp_cov(self):
        """Test the interpolation between time steps"""
        target_time = 0.0
        interp_mns, interp_covs = gf.interp_cov(target_time, self.star_pars)
        self.assertTrue(np.allclose(interp_mns, self.xyzuvw[:,0,:]))
        self.assertTrue(np.allclose(interp_covs, self.xyzuvw_cov[:,0]))
    
        target_time = 1.0
        interp_mns, interp_covs = gf.interp_cov(target_time, self.star_pars)
        self.assertTrue(np.allclose(interp_mns, self.xyzuvw[:,1,:]))
        self.assertTrue(np.allclose(interp_covs, self.xyzuvw_cov[:,1]))

        target_time = 0.5
        interp_mns, interp_covs = gf.interp_cov(target_time, self.star_pars)
        # check covariance matrices have 1.5 and 3.5 along diagonals
        self.assertTrue(np.allclose(interp_covs[0], np.eye(6)*1.5))
        self.assertTrue(np.allclose(interp_covs[1], np.eye(6)*3.5))

        # first star mean should be average between timesteps
        self.assertTrue(
            np.allclose(
                (self.xyzuvw_cov[0,0] + self.xyzuvw_cov[0,1])/2.0,
                interp_covs[0]
            )
        )

        # second star should show no interpolation in mean
        self.assertTrue(np.allclose(self.xyzuvw[1,0], interp_mns[1]))

    def find_nearest(self,array,value):
        """Basic tool to find the array element that is closest to a value"""
        idx = (np.abs(array-value)).argmin()
        return array[idx]

    def test_eig_prior(self):
        """Test that the eig_prior is maximal at the characteristic minima"""
        eig_vals = np.linspace(0.1,20,200)
        char_mins = np.array([0.5, 1.0, 2.0, 4.0, 8.0])

        for ctr, char_min in enumerate(char_mins):
            self.assertEqual(
                eig_vals[np.argmax(gf.eig_prior(char_min, eig_vals))],
                self.find_nearest(eig_vals,char_min),
                msg="{}...".format(ctr),
            )

    def test_lnprior(self):
        """Test correctness of lnprior - only the final set of pars is valid"""
        many_group_pars = np.array([
            # X, Y, Z, U,  V,  W,1/dX,1/dY,1/dZ,1/dV,Cxy,Cxz,Cyz,age,nstars
            [ 0, 0, 0, 0,  0,  0,1/10,1/10,1/10, 1/5, .5, .2, .3,-20, 100],
            [20,20,20, 5,1e9,  5,1/10,1/10,1/10, 1/5,-.3,-.6, .2,  1, 100],
            [50,50,50, 0,  0,-10,1/10,1/10,1/10, 1/5,1.8, .3, .2,  1, 100],
            [50,50,50, 0,  0,-10,1/10,1/-5,1/10, 1/5,-.8, .3, .2,  1, 100],
            [50,50,50, 0,  0,-10,1/10,1/10,1/10, 1/5,-.8, .3, .2, 10, 100],
            [ 0, 0, 0, 0,  0,  0,1/10,1/10,1/10, 1/5, .5, .2, .3,  1, 100],
        ])
        
        priors = np.zeros(many_group_pars.shape[0])

        for i in range(many_group_pars.shape[0]):
            priors[i] = gf.lnprior(many_group_pars[i], None, self.star_pars)

        self.assertEqual(np.max(priors[:-2]), -np.inf)
        self.assertEqual(priors[-1], 0.0)

    def test_lnlike(self):
        """
        Compare likelihood results for the groups and given stars
        """
        # at t = 0.0 both stars are at around
        # [0.0,0.5,1.0,1.0,0.2,-1.0],

        z = np.ones(self.xyzuvw.shape[0])

        group_pars1 = np.array(
            [0.0,0.5,0.1,1.0,0.2,-1.0,1/5,1/5,1/5,1/2,0,0,0,0.0]
        )
        # same as 1 but double the width
        group_pars2 = np.array(
            [0.0,0.5,0.1,1.0,0.2,-1.0,1/10,1/10,1/10,1/4,0,0,0,0.0]
        )
        group_pars3 = np.array(
            [0.0,0.5,0.1,1.0,0.2,-1.0,1/5,1/5,1/5,1/2,0,0,0,0.5]
        )
        group_pars4 = np.array(
            [100,100,100,50,50,50,1/10,1/10,1/10,1/4,0,0,0,0.0]
        )

        # assert 1 > [2,3,4], [2,3] > 4
        self.assertTrue(
            self.lnlike(group_pars1,z,self.star_pars) > 
            self.lnlike(group_pars2,z,self.star_pars)
        )
        self.assertTrue(
            self.lnlike(group_pars1,z,self.star_pars) > 
            self.lnlike(group_pars3,z,self.star_pars)
        )
        self.assertTrue(
            self.lnlike(group_pars1,z,self.star_pars) > 
            self.lnlike(group_pars4,z,self.star_pars)
        )
        self.assertTrue(
            self.lnlike(group_pars2,z,self.star_pars) > 
            self.lnlike(group_pars4,z,self.star_pars)
        )
        self.assertTrue(
            self.lnlike(group_pars3,z,self.star_pars) > 
            self.lnlike(group_pars4,z,self.star_pars)
        )
        
        return 0

    def test_fit_group(self):
        """
        Synthesise a tb file with negligible error, retrieve initial
        parameters
        """
        # Not ready to test this yet
        self.assertTrue(False)
        
        # neligible error, anything smaller runs into problems with matrix
        # inversions
        error = 1e-5
        ntimes = 20
        
        syn.synthesise_data(1,self.group_pars,error,savefile=self.synth_file)

        with open(self.synth_file, 'r') as fp:
            t = pickle.load(fp)

        times = np.linspace(0,self.group_pars[-2],ntimes)
        tb.traceback(t,times,savefile=self.tb_file)

        best_fit, memb  = gf.fit_group(self.tb_file)

        # check membership list totals to nstars in group
        self.assertEqual(int(round(np.sum(memb))), self.group_pars[-1])
        self.assertEqual(round(np.max(memb)), 1.0)
        self.assertEqual(round(np.min(memb)), 1.0)

        ctr = 0 # left here for convenience... tidy up later

        means = best_fit[0:6]
        stds  = best_fit[6:10]
        corrs = best_fit[10:13]
        age   = best_fit[13]

        tol_mean = 3.5
        tol_std  = 2.5
        tol_corr = 0.3
        tol_age  = 0.5

        self.assertTrue(
            np.max(abs(means - self.group_pars[0:6])) < tol_mean,
            msg="\nFailed {} received:\n{}\nshould be within {} to:\n{}".\
            format(ctr, means, tol_mean, self.group_pars[0:6]))
        self.assertTrue(
            np.max(abs(stds - self.group_pars[6:10])) < tol_std,
            msg="\nFailed {} received:\n{}\nshould be close to:\n{}".\
            format(ctr, stds, tol_std, self.group_pars[6:10]))
        self.assertTrue(
            np.max(abs(corrs - self.group_pars[10:13])) < tol_corr,
            msg="\nFailed {} received:\n{}\nshould be close to:\n{}".\
            format(ctr, corrs, tol_corr, self.group_pars[10:13]))
        self.assertTrue(
            np.max(abs(age - self.group_pars[13])) < tol_age,
            msg="\nFailed {} received:\n{}\nshould be close to:\n{}".\
            format(ctr, age, tol_age, self.group_pars[13]))

