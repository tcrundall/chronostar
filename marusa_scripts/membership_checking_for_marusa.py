# coding: utf-8

import numpy as np
import sys
import chronostar.retired2.datatool as dt

#~ sys.path.insert(0, '/home/marusa/chronostar/')

root = sys.argv[1]
print(root)

#~ fmerrs = np.load('final_med_errs.npy')
#~ fmerrs.shape
#~ fmerrs[:,-1]
z = np.load(os.path.join(root, 'final_membership.npy'))
print z.shape
print z.sum(axis=0)



#~ star_pars = dt.loadDictFromTable('/home/marusa/chronostar/data/marusa_galah_li_strong_stars_xyzuvw.fits')
#~ print star_pars.keys()
#~ print star_pars['table'].info
#~ print star_pars['indices'].shape
#~ print len(star_pars['indices'])
#~ print len(star_pars['table'])
#~ print z.shape
#~ print comp_1_table = star_pars['table'][np.where(z[:,0] > 0.5)]
#~ print comp_1_table
