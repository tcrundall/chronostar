# coding: utf-8
get_ipython().magic(u'ls ')
import numpy as np
z = np.load('final_membership.npy')
fmerrs = np.load('final_med_errs.npy')
fmerrs
fmerrs[0]
import sys
sys.path.insert(0, '/home/tcrun/chronostar/')
import chronostar.retired2.datatool as dt
sp = dt.loadDictFromTable('../../../data/marusa_galah_li_strong_stars_xyzuvw.fits')
get_ipython().magic(u'ls ')
sp['table'].info
comp_a = sp['table'][np.where(z[:,0] > 0.5)]
len(comp_a)
comp_a['X']
comp_a['U']
comp_a['V']
import matplotlib.pyplot as plt
plt.plot(comp_a['X'], comp_a['U'], '.')
plt.show()
plt.clf()
plt.plot(comp_a['U'], comp_a['V'], '.')
plt.show()
np.mean(comp_a['U'])
np.mean(comp_a['V'])
import chronostar.coordinate as coord
coord.lsr
u_mean = np.mean(comp_a['U'])
v_mean = np.mean(comp_a['V'])
xyzuvw = sp['xyzuvw'][np.where(z[:,0] > 0.5)]
xyzuvw.shape
np.mean(xyzuvw, axis=0)
mean_lsr = np.mean(xyzuvw, axis=0)
mean_hc = coord.convert_lsr2helio(mean_lsr)
mean_hc
