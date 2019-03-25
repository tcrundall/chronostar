# coding: utf-8
fmerrs = np.load('final_med_errs.npy')
import numpy as np
fmerrs = np.load('final_med_errs.npy')
fmerrs.shape
fmerrs[:,-1]
z = np.load('final_membership.npy')
z.shape
z.sum(axis=0)
import sys
sys.path.insert(0, '/home/marusa/chronostar/')
import chronostar.retired2.datatool as dt
star_pars = dt.loadDictFromTable('../../../../../../data/marusa_galah_li_strong_stars_xyzuvw.fits')
star_pars.keys()
star_pars['table'].info
star_pars['indices'].shape
len(star_pars['indices'])
len(star_pars['table'])
z.shape
comp_1_table = star_pars['table'][np.where(z[:,0] > 0.5)]
comp_1_table
